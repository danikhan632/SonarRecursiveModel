"""
Simple SONAR encoder/decoder using transformers library.
Lighter weight alternative to fairseq implementation.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

class SimpleSonarEncoder:
    """
    SONAR text encoder using transformers library.
    """
    def __init__(self, model_name="cointegrated/SONAR_200_text_encoder", device="cuda"):
        self.device = device
        self.encoder = M2M100Encoder.from_pretrained(model_name).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, texts, lang='eng_Latn', norm=False):
        """
        Encode texts to SONAR embeddings.

        Args:
            texts: List of strings or single string
            lang: Language code (default: eng_Latn)
            norm: Whether to L2 normalize embeddings

        Returns:
            Tensor of shape [batch_size, 1024]
        """
        if isinstance(texts, str):
            texts = [texts]

        self.tokenizer.src_lang = lang

        with torch.inference_mode():
            batch = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.device)
            seq_embs = self.encoder(**batch).last_hidden_state
            mask = batch.attention_mask

            # Mean pooling
            mean_emb = (seq_embs * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)

            if norm:
                mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=-1)

        return mean_emb

class SimpleSonarDecoder:
    """
    SONAR decoder using transformers library.
    Note: Decoding from embeddings is more complex and may require specialized models.
    This is a placeholder for when such models become available.
    """
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", device="cuda"):
        self.device = device
        # For now, using NLLB as it's architecturally similar
        # A proper SONAR decoder would be trained specifically for embedding-to-text
        print("Warning: Using NLLB as decoder proxy. For true SONAR decoding, use fairseq implementation.")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def decode(self, embeddings, target_lang='eng_Latn', max_length=512):
        """
        Decode embeddings back to text.

        Args:
            embeddings: Tensor of shape [batch_size, 1024]
            target_lang: Target language code
            max_length: Maximum generation length

        Returns:
            List of decoded strings
        """
        # Note: This is a simplified approach
        # True SONAR decoding requires a model trained to decode from embeddings
        print("Warning: Decoding from raw embeddings not fully supported yet.")
        print("Consider using fairseq SONAR decoder for production use.")

        # Placeholder: return empty strings
        batch_size = embeddings.shape[0]
        return ["[Decoding not implemented - use fairseq decoder]"] * batch_size

# Convenience functions
def encode_texts(texts, lang='eng_Latn', norm=False, device='cuda'):
    """
    Quick function to encode texts to SONAR embeddings.

    Args:
        texts: List of strings or single string
        lang: Language code
        norm: Whether to normalize
        device: cuda or cpu

    Returns:
        Tensor of embeddings [batch_size, 1024]
    """
    encoder = SimpleSonarEncoder(device=device)
    return encoder.encode(texts, lang=lang, norm=norm)

if __name__ == "__main__":
    # Test the encoder
    print("Testing Simple SONAR Encoder...")

    encoder = SimpleSonarEncoder(device="cuda" if torch.cuda.is_available() else "cpu")

    sentences = [
        'My name is SONAR.',
        'I can embed the sentences into vectorial space.'
    ]

    embs = encoder.encode(sentences, lang="eng_Latn")
    print(f"Embeddings shape: {embs.shape}")  # Should be [2, 1024]
    print(f"First few values: {embs[0, :5]}")

    # Test single sentence
    single_emb = encoder.encode("This is a test sentence.")
    print(f"Single embedding shape: {single_emb.shape}")  # Should be [1, 1024]

    print("\nEncoder test complete!")
