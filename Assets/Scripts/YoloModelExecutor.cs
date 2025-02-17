using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using Unity.Sentis;
using UnityEngine;

namespace Assets.Scripts
{
    /// <summary>
    ///     Model executor using a YOLO object detection model (with optional mock results).
    /// </summary>
    public class YoloModelExecutor : MonoBehaviour
    {
        /// <summary>
        ///     Object detection model that should be executed (only used if mockResults == false).
        /// </summary>
        public ModelAsset ModelAsset;

        /// <summary>
        ///     Material with shader that is used for scaling the input image to the correct aspect ratio.
        /// </summary>
        public Material ShaderForScaling;

        private TextureTransform textureTransform;
        private bool disposed;
        private IWorker worker;
        private TensorFloat inputTensor;
        private TensorFloat outputTensor;
        private ModelState modelState = ModelState.PreProcessing;
        private CameraTransform cameraTransform;
        private bool hasMoreModelToRun = true;
        private IEnumerator modelEnumerator;
        private RenderTexture intermediateRenderTexture;
        private YoloDebugOutput yoloDebugOutput;
        private int layerCount;
        private float threshold;
        private YoloRecognitionHandler yoloRecognitionHandler;

        // NEW: Toggle to skip real inference and use mock detections
        // [SerializeField]
        private bool mockResults = true;

        // NEW: Store mocked detections after "Executing"
        private List<YoloItem> mockDetections;

        private static WebCamTextureAccess WebCamTextureAccess => WebCamTextureAccess.Instance;
        private static SettingsProvider SettingsProvider => SettingsProvider.Instance;

        private void Start()
        {
            // Get other components
            this.yoloDebugOutput = gameObject.GetComponent<YoloDebugOutput>();
            this.yoloRecognitionHandler = gameObject.GetComponent<YoloRecognitionHandler>();

            // Initialize settings
            this.SettingsProviderOnPropertyChanged(null, new PropertyChangedEventArgs(nameof(SettingsProvider.ModelExecutionOR)));
            this.SettingsProviderOnPropertyChanged(null, new PropertyChangedEventArgs(nameof(SettingsProvider.ThresholdOR)));
            SettingsProvider.PropertyChanged += this.SettingsProviderOnPropertyChanged;

            // We still initialize the webcam (if needed for debug visuals)
            WebCamTextureAccess.Play();

            // Create intermediate RenderTexture
            this.intermediateRenderTexture = new RenderTexture(Parameters.ModelImageResolution.x, Parameters.ModelImageResolution.y, 24);
            this.ShaderForScaling.SetFloat("_Aspect",
                (float)WebCamTextureAccess.ActualCameraSize.x / WebCamTextureAccess.ActualCameraSize.y
                * Parameters.ModelImageResolution.y / Parameters.ModelImageResolution.x);

            this.textureTransform = new TextureTransform().SetDimensions(
                Parameters.ModelImageResolution.x,
                Parameters.ModelImageResolution.y,
                3);

            // NEW/CHANGED: Only load and create the worker if we are NOT mocking results
            if (!this.mockResults)
            {
                // Load the model from the provided NNModel asset
                Model model = ModelLoader.Load(this.ModelAsset);

                // Create a Barracuda worker to run the model on the GPU
                this.worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
            }
        }

        private void Update()
        {
            switch (this.modelState)
            {
                case ModelState.Idle:
                    // Do nothing
                    break;

                case ModelState.PreProcessing:
                    this.inputTensor?.Dispose();

                    // Capture current camera transform
                    this.cameraTransform = new CameraTransform(Camera.main);

                    // Blit the webcam feed into a RT with the correct aspect ratio
                    Graphics.Blit(WebCamTextureAccess.WebCamTexture, this.intermediateRenderTexture, this.ShaderForScaling);

                    // Convert the RT into a TensorFloat
                    this.inputTensor = TextureConverter.ToTensor(this.intermediateRenderTexture, this.textureTransform);

                    // Move to the 'Executing' state
                    this.modelState = ModelState.Executing;
                    break;

                case ModelState.Executing:
                    // NEW/CHANGED: If we're mocking results, skip real inference
                    if (!this.mockResults)
                    {
                        this.modelEnumerator ??= this.worker.StartManualSchedule(this.inputTensor);

                        int i = 0;
                        while (i++ < this.layerCount && this.hasMoreModelToRun)
                        {
                            this.hasMoreModelToRun = this.modelEnumerator.MoveNext();
                        }

                        if (!this.hasMoreModelToRun)
                        {
                            // reset model states
                            this.modelEnumerator = null;
                            this.hasMoreModelToRun = true;
                            this.modelState = ModelState.ReadOutput;
                        }
                    }
                    else
                    {
                        // If mocking, just skip ahead to "ReadOutput"
                        this.modelState = ModelState.ReadOutput;
                    }

                    break;

                case ModelState.ReadOutput:
                    if (!this.mockResults)
                    {
                        // Normal pipeline: read from the real model output
                        this.outputTensor = (TensorFloat)this.worker.PeekOutput();
                        this.modelState = ModelState.Idle;

                        // Wait for async GPU readback
                        this.outputTensor.AsyncReadbackRequest(_ => this.modelState = ModelState.PostProcessing);
                    }
                    else
                    {
                        // NEW/CHANGED: Create mock detections
                        this.mockDetections = CreateMockDetections();

                        // Immediately jump to post-processing
                        this.modelState = ModelState.PostProcessing;
                    }
                    break;

                case ModelState.PostProcessing:
                    // Interpret the model output (or mocked output)
                    List<YoloItem> result;
                    if (!this.mockResults)
                    {
                        // Make the real output CPU-readable
                        this.outputTensor.MakeReadable();

                        // Run your normal output processing
                        result = YoloModelOutputProcessor.ProcessModelOutput(this.outputTensor, this.threshold);
                    }
                    else
                    {
                        // If mocking, we already have a list of items
                        result = this.mockDetections;
                    }

                    // Display debug info
                    this.yoloDebugOutput.ShowDebugInformation(this.inputTensor, result, this.cameraTransform);

                    // Handle recognized objects
                    this.yoloRecognitionHandler.ShowRecognitions(result, this.cameraTransform);

                    // Reset state machine for the next frame
                    this.modelState = ModelState.PreProcessing;
                    break;

                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        /// <summary>
        ///     Method that is called when the object is destroyed.
        /// </summary>
        public void OnDestroy()
        {
            if (this.disposed)
            {
                return;
            }

            this.disposed = true;
            SettingsProvider.PropertyChanged -= this.SettingsProviderOnPropertyChanged;
            WebCamTextureAccess.Stop();

            this.inputTensor?.Dispose();

            // NEW/CHANGED: Only dispose the worker if we're actually using one
            if (!this.mockResults && this.worker != null)
            {
                this.worker.Dispose();
            }

            // Release and destroy the intermediate RenderTexture
            if (this.intermediateRenderTexture != null)
            {
                this.intermediateRenderTexture.Release();
                Destroy(this.intermediateRenderTexture);
            }
        }

        private void SettingsProviderOnPropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            switch (e.PropertyName)
            {
                case nameof(SettingsProvider.ModelExecutionOR):
                    this.UpdateModelPerformance();
                    break;
                case nameof(SettingsProvider.ThresholdOR):
                    this.UpdateThreshold();
                    break;
            }
        }

        private void UpdateModelPerformance()
        {
            this.layerCount = SettingsProvider.ModelExecutionOR switch
            {
                ModelExecutionMode.High => Parameters.LayersHigh,
                ModelExecutionMode.Low => Parameters.LayersLow,
                ModelExecutionMode.Full => int.MaxValue,
                _ => this.layerCount
            };
        }

        private void UpdateThreshold()
        {
            this.threshold = SettingsProvider.ThresholdOR switch
            {
                RecognitionThreshold.High => Parameters.ThresholdHigh,
                RecognitionThreshold.Medium => Parameters.ThresholdMedium,
                RecognitionThreshold.Low => Parameters.ThresholdLow,
                _ => this.threshold
            };
        }

        // NEW: A helper method to create mock detections
        private List<YoloItem> CreateMockDetections()
        {
            List<YoloItem> detections = new List<YoloItem>();

            // Grab the model's expected width/height
            float width = Parameters.ModelImageResolution.x;
            float height = Parameters.ModelImageResolution.y;

            // Decide how big you want the bounding box
            float boxWidth = 100f;
            float boxHeight = 100f;

            // Compute the center point
            Vector2 center = new Vector2(width / 2f, height / 2f);

            // Compute top-left and bottom-right based on the box size
            Vector2 topLeft = center - new Vector2(boxWidth / 2f, boxHeight / 2f);
            Vector2 bottomRight = center + new Vector2(boxWidth / 2f, boxHeight / 2f);

            // Confidence and class index are arbitrary
            float confidence = 0.95f;
            int classIndex = 0;

            // If your pipeline expects YOLOv10 style bounding boxes:
            var item = YoloItem.FromVersion10(topLeft, bottomRight, confidence, classIndex);

            detections.Add(item);

            return detections;
        }
    }
}
