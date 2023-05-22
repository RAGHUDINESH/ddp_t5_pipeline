from transformers import T5ForConditionalGeneration, AutoTokenizer
import json
def load_t5_inline(model="small", device = "cpu"):
  t5 = f"t5-{model}"
  model = T5ForConditionalGeneration.from_pretrained(t5)
  model = model.to(device)
  tokenizer = AutoTokenizer.from_pretrained(t5, model_max_length = model.config.n_positions)
  with open("data/ner_dict.json") as f:
    ner_dict = json.load(f)
  space_token = list(tokenizer.get_vocab().keys())[list(tokenizer.get_vocab().values()).index(3)]
  add_tokens = [space_token+token for token in ner_dict.values()]
  tokenizer.add_tokens(add_tokens)
  return model,tokenizer