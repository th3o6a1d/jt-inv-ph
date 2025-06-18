from transformers import DistilBertModel, DistilBertPreTrainedModel
import torch.nn as nn
import torch

class DistilBertForICD10(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:,0]  # CLS token
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        return {"loss": loss, "logits": logits}

def get_model(num_labels):
    from transformers import DistilBertConfig
    config = DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    return DistilBertForICD10(config) 