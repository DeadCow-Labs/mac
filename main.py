from flask import Flask, request, jsonify
import os
import torch
from concrete.ml.sklearn import LogisticRegression
from torch.quantization import quantize_dynamic
import numpy as np
from transformers import BertModel, BertTokenizer
# Hypothetical FHE library import
# from fhe_library import FHEContext, encrypt, decrypt

app = Flask(__name__)

# Global variable to store the compiled model
compiled_model = LogisticRegression(n_bits=8)  # Replace with actual initialization

# Load the BERT model and tokenizer
model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Initialize FHE context (hypothetical)
# fhe_context = FHEContext()

@app.route('/nodes/upload_model', methods=['POST'])
def upload_model():
    model_file = request.files.get('model_file')
    name = request.form.get('name')
    owner_address = request.form.get('owner_address')

    if not model_file:
        return jsonify({"error": "Model file not provided"}), 400

    try:
        # Save the model file
        model_path = os.path.join(os.getcwd(), model_file.filename)
        model_file.save(model_path)
        print(f"Model {model_file.filename} received and saved to {model_path}.")

        # No loading or processing of the model
        return jsonify({"status": "success", "message": f"Model {model_file.filename} saved successfully."}), 200
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return jsonify({"error": f"Error saving model: {str(e)}"}), 500

# @app.route('/nodes/run_inference', methods=['POST'])
# def run_inference():
#     global compiled_model
#     if compiled_model is None:
#         return jsonify({"error": "Model not loaded"}), 500

#     try:
#         # Receive encrypted input data
#         request_data = request.get_json()
#         encrypted_input = request_data.get("encrypted_input")

#         if not encrypted_input:
#             return jsonify({"error": "Encrypted input data not provided"}), 400

#         # Perform inference on the encrypted input
#         encrypted_output = compiled_model.fhe_circuit.run(encrypted_input)

#         # Return the encrypted output
#         return jsonify({"encrypted_output": encrypted_output}), 200
#     except Exception as e:
#         print(f"Error during inference: {str(e)}")
#         return jsonify({"error": f"Error during inference: {str(e)}"}), 500


# Global variable to store the compiled model
compiled_model = None

# Function to inspect the state dictionary
def inspect_state_dict(file_path):
    try:
        state_dict = torch.load(file_path)
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"Layer: {key}, Shape: {value.shape}")
            else:
                print(f"Layer: {key}, Type: {type(value)}")
    except Exception as e:
        print(f"Failed to load state dictionary: {str(e)}")

def find_and_load_model():
    global compiled_model
    current_directory = os.getcwd()
    
    model_filename = 'tmpzg6l206p'  # Use the actual filename

    file_path = os.path.join(current_directory, model_filename)
    if os.path.isfile(file_path):
        try:
            # Initialize a pre-trained model class
            model = BertModel.from_pretrained('bert-base-uncased')

            # Load the state dictionary
            state_dict = torch.load(file_path)
            model.load_state_dict(state_dict, strict=False)
            model.eval()  # Set the model to evaluation mode

            compiled_model = model
            print(f"Model loaded successfully from {file_path}")
            return
        except Exception as e:
            print(f"Failed to load model from {file_path}: {str(e)}")
    raise FileNotFoundError("No valid model file found in the current directory")



@app.route('/nodes/run_inference', methods=['POST'])
def run_inference():
    try:
        # Receive input data
        request_data = request.get_json()
        input_text = request_data.get("input_data")

        if not input_text:
            return jsonify({"error": "Input data not provided"}), 400

        # Tokenize input text
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        # Encrypt input data (hypothetical)
        # encrypted_input = encrypt(fhe_context, input_ids)

        # Perform inference
        with torch.no_grad():
            # Assuming the model can handle encrypted input directly
            # output = model(encrypted_input)
            output = model(input_ids)

        # Access the desired part of the output
        cls_token_output = output.last_hidden_state[:, 0, :]

        # Decrypt output data (hypothetical)
        # decrypted_output = decrypt(fhe_context, cls_token_output)

        # Convert the result to a list
        result_list = cls_token_output.squeeze().tolist()  # Use decrypted_output if applicable

        return jsonify({"result": result_list}), 200
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return jsonify({"error": f"Error during inference: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)