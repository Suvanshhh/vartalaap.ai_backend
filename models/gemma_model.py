import keras
import keras_nlp
import tensorflow as tf
import os

class GemmaChatModel:
    def __init__(self, weights_path=None):
        self.model = None
        self.sampler = None
        self.initialize_model(weights_path)

    def initialize_model(self, weights_path=None):
        """Initialize the Gemma model with LoRA and load weights"""
        try:
            # Initialize base model
            print("Loading base Gemma model...")
            self.model = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_2b_en")
            
            # Enable LoRA
            print("Enabling LoRA...")
            self.model.backbone.enable_lora(rank=8)
            
            # Load LoRA weights if provided
            if weights_path and os.path.exists(weights_path):
                print(f"Loading LoRA weights from {weights_path}")
                try:
                    self.model.backbone.load_lora_weights(weights_path)
                    print("LoRA weights loaded successfully")
                except Exception as e:
                    print(f"Error loading LoRA weights: {str(e)}")
            else:
                print("LoRA weights file not found. Continuing with base model.")

            # Configure model parameters
            self.model.preprocessor.sequence_length = 256
            self.sampler = keras_nlp.samplers.TopKSampler(k=5, temperature=0.7)
            self.model.compile(sampler=self.sampler)
            print("Gemma model initialized successfully.")
        
        except Exception as e:
            print(f"Error initializing the Gemma model: {str(e)}")
            self.model = None  # Ensure model is set to None if initialization fails

    def generate_response(self, user_input):
        """Generate response for user input"""
        if not self.model:
            print("Model is not loaded correctly. Cannot generate response.")
            return "I apologize, but I'm having trouble generating a response right now."
        
        try:
            prompt = f"Instruction:\n{user_input}\n\nResponse:\n"
            response = self.model.generate(prompt, max_length=256)
            
            # Extract the response part
            response_text = response.split("Response:\n")[-1].strip()
            
            return response_text
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now."

# Initialize model with weights
WEIGHTS_PATH = "path/to/your/Gemma_LoRA_finetuned_weights.h5"
gemma_model = GemmaChatModel(weights_path=WEIGHTS_PATH)



# import keras_nlp
# import tensorflow as tf

# try:
#     print("Loading Gemma model...")
#     gemma_model = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_2b_en")
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading Gemma model: {e}")

# import tensorflow as tf
# import keras
# import keras_nlp

# print("TensorFlow version:", tf.__version__)
# print("Keras version:", keras.__version__)
# print("Keras NLP version:", keras_nlp.__version__)
