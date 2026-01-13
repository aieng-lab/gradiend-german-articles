import torch
from transformers import AutoModelForMaskedLM, BertTokenizer
from gradiend.training import ModelWithGradiend



def evaluate(model, tokenizer, masked_text, masked_word=None, top_k=25):
    """
    Evaluate the model on masked language modeling (MLM) task.

    Args:
    - model: the transformer model
    - tokenizer: The BERT tokenizer.
    - masked_text: The text with a masked token (e.g., "The capital of France is [MASK].").
    - masked_word: The original word that was masked (for reference).
    - top_k: The number of top predictions to display.
    """
    # Tokenize the input text
    inputs = tokenizer(masked_text, return_tensors="pt")
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the index of the masked token
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Pass the inputs through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the logits and softmax to get probabilities
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
        mask_token_logits = logits[0, mask_token_index, :]
        probabilities = torch.softmax(mask_token_logits, dim=-1)
    else:
        probabilities = outputs[0, mask_token_index, :]

    # Get the top k predictions
    top_k_tokens = torch.topk(probabilities, top_k, dim=1)
    top_k_token_indices = top_k_tokens.indices[0].tolist()
    top_k_token_probabilities = top_k_tokens.values[0].tolist()

    # Decode the predicted tokens
    predictions = [tokenizer.decode([token]) for token in top_k_token_indices]


    # Print the results with probabilities
    print(f"Masked text: {masked_text}")
    if masked_word:
        print(f"Original word: {masked_word}")
    print(f"Top {top_k} predictions:")
    for i, (prediction, probability) in enumerate(zip(predictions, top_k_token_probabilities), 1):
        print(f"{i}: {prediction} (Probability: {probability:.4f})")



if __name__ == '__main__':
    masked_text = "Eva is traveling. [MASK] reads a book."
    masked_text = "Paul was working. [MASK] coded in Python."
    masked_text = "My friend [MASK] works as a software engineer"
    masked_text = "My friend [MASK] works as a social worker"

    model_name = 'bert-base-uncased'
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    ae_model = 'results/models/ae17_tanh_target_diff_lr3_best'
    bert_with_ae = ModelWithGradiend.from_pretrained(ae_model)
    enhanced_model = bert_with_ae.modify_model(feature_factor=0, lr=-0.013)

    masked_word = "He"
    evaluate(model, tokenizer, masked_text, masked_word)
    evaluate(enhanced_model, tokenizer, masked_text, masked_word)
