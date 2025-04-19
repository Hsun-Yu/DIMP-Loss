
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import BertPreTrainedModel, BertModel, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class FocalLoss(nn.Module):
    """
    Focal Loss
    
    this class is copy from https://github.com/clcarwin/focal_loss_pytorch
    """
    def __init__(self, gamma=1, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1 - alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        # print("OLD")
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        # p = (1-pt)
        # p = (p - p.min()) / (p.max() - p.min())
        
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
class MyFocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=None, size_average=True, version=2):
        """
        Custom implementation of Focal Loss with multiple versions.

        Parameters:
        - gamma: Focusing parameter to adjust the rate at which easy examples are down-weighted.
        - alpha: Class balancing factor. Can be a float, int, or list for multi-class balancing.
        - size_average: Whether to average the loss over all samples.
        - version: Determines the specific version of the loss function to use.
          - Version 1: loss = -alpha * (1 - pt + pt2)**gamma * log(pt)
          - Version 2 (DIMP-Loss): loss = -alpha * (pt2/pt)**gamma * log(pt)
          - Version 3: loss = -alpha * (pt2/pt3)**gamma * log(pt)
          - Version 4: loss = -alpha * (pt2)**gamma * log(pt)
        """
        super(MyFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1 - alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        
        self.version = version

    def forward(self, input, target, input2=None, input3=None):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
            
            input2 = input2.view(input2.size(0), input2.size(1), -1)  # N,C,H,W => N,C,H*W 
            input2 = input2.transpose(1, 2)
            input2 = input2.contiguous().view(-1, input2.size(2))
            
            if self.version == 3:
                input3 = input3.view(input3.size(0), input3.size(1), -1)
                input3 = input3.transpose(1, 2)
                input3 = input3.contiguous().view(-1, input3.size(2))
                
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        
        # print("input2", input2)
        logpt2 = F.log_softmax(input2)
        logpt2 = logpt2.gather(1, target)
        logpt2 = logpt2.view(-1)
        pt2 = Variable(logpt2.data.exp())
        
        if self.version == 3:
            logpt3 = F.log_softmax(input3)
            logpt3 = logpt3.gather(1, target)
            logpt3 = logpt3.view(-1)
            pt3 = Variable(logpt3.data.exp())
        
        if self.version == 2:
            p = pt2 / pt
        elif self.version == 3:
            p = pt2 / pt3
        elif self.version == 4:
            p = pt2

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        if self.version == 1:
            loss = -1 * (1-pt+pt2)**self.gamma * logpt
        elif self.version == 2:
            loss = -1 * (p)**self.gamma * logpt
        elif self.version == 3:
            loss = -1 * (p)**self.gamma * logpt
        elif self.version == 4:
            loss = -1 * (p)**self.gamma * logpt
        
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
class MyNewBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.tokenizer = None
        
        self.model2 = None
        self.model2_tokenizer = None
        self.model3 = None
        self.model3_tokenizer = None
        
        # for contrastive learning
        self.contrastive_learning = None
        self.temperature = None
        self.contrastive_learning_weight = None
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.model2 is not None:
            # This is for DIMP-Loss's quality checker
            self.model2.eval()
            with torch.no_grad():
                outputs2 = self.model2(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        
        if self.model3 is not None:
            # This is for IMP-Loss's diversity checker
            self.model3.eval()
            with torch.no_grad():
                outputs3 = self.model3(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # If contrastive learning is enabled, compute the contrastive loss using dropout twice
        if self.contrastive_learning:
            pooled_output_1 = self.dropout(pooled_output)  # Dropout for the first view
            pooled_output_2 = self.dropout(pooled_output)  # Dropout for the second view
            contrastive_loss = self.compute_contrastive_loss(pooled_output_1, pooled_output_2, self.temperature)
        else:
            contrastive_loss = None

        loss = None
        # Run for loss calculation
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            
            # print("self.config.problem_type", self.config.problem_type)
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "single_label_classification_myloss_v1":
                loss_fct = MyFocalLoss(version=1)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), outputs2.logits.view(-1, self.num_labels))
            elif self.config.problem_type == "single_label_classification_myloss_v2":
                # DIMP-Loss
                loss_fct = MyFocalLoss(version=2)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), outputs2.logits.view(-1, self.num_labels))
            elif self.config.problem_type == "single_label_classification_myloss_importance":
                loss_fct = MyFocalLoss(version=3)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), outputs2.logits.view(-1, self.num_labels), outputs3.logits.view(-1, self.num_labels))
            elif self.config.problem_type == "single_label_classification_distillation":
                loss_fct = MyFocalLoss(version=4)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), outputs2.logits.view(-1, self.num_labels))
            elif self.config.problem_type == "single_label_classification_focalloss":
                # For focal loss
                loss_fct = FocalLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                
        # Combine losses if both are present
        if loss is not None and contrastive_loss is not None:
            loss += contrastive_loss * self.contrastive_learning_weight
        elif contrastive_loss is not None:
            loss = contrastive_loss
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def compute_contrastive_loss(self, pooled_output_1: torch.Tensor, pooled_output_2: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Computes contrastive loss for SimCSE using two different dropout views.
        """
        batch_size = pooled_output_1.size(0)

        # Normalize the pooled outputs for contrastive loss
        pooled_output_1 = F.normalize(pooled_output_1, p=2, dim=1)
        pooled_output_2 = F.normalize(pooled_output_2, p=2, dim=1)

        # Compute similarity matrix between the two views
        similarity_matrix = torch.matmul(pooled_output_1, pooled_output_2.T) / temperature

        # Create labels for contrastive learning
        labels = torch.arange(batch_size).to(pooled_output_1.device)

        # Compute cross-entropy loss
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(similarity_matrix, labels)
        return loss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class MyNewRobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.classifier = RobertaClassificationHead(config)
        self.tokenizer = None
        
        self.model2 = None
        self.model2_tokenizer = None
        self.model3 = None
        self.model3_tokenizer = None
        
        # for contrastive learning
        self.contrastive_learning = None
        self.temperature = None
        self.contrastive_learning_weight = None
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # prepare input for model2 and model3
        decoded_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        if self.model2 is not None:
            model2_inputs = self.model2_tokenizer.batch_encode_plus(decoded_inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
            self.model2.eval()
            with torch.no_grad():
                outputs2 = self.model2(
                    **model2_inputs
                )
        
        if self.model3 is not None:
            model3_inputs = self.model3_tokenizer.batch_encode_plus(decoded_inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
            self.model3.eval()
            with torch.no_grad():
                outputs3 = self.model3(
                    **model3_inputs
                )
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        
        # If contrastive learning is enabled, compute the contrastive loss using dropout twice
        if self.contrastive_learning:
            pooled_output_1 = self.dropout(pooled_output)  # Dropout for the first view
            pooled_output_2 = self.dropout(pooled_output)  # Dropout for the second view
            contrastive_loss = self.compute_contrastive_loss(pooled_output_1, pooled_output_2, self.temperature)
        else:
            contrastive_loss = None

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            
            # print("self.config.problem_type", self.config.problem_type)
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # CE-Loss
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "single_label_classification_myloss_v1":
                loss_fct = MyFocalLoss(version=1)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), outputs2.logits.view(-1, self.num_labels))
            elif self.config.problem_type == "single_label_classification_myloss_v2":
                # DIMP-Loss
                loss_fct = MyFocalLoss(version=2)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), outputs2.logits.view(-1, self.num_labels))
            elif self.config.problem_type == "single_label_classification_myloss_importance":
                # IMP-Loss
                loss_fct = MyFocalLoss(version=3)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), outputs2.logits.view(-1, self.num_labels), outputs3.logits.view(-1, self.num_labels))
            elif self.config.problem_type == "single_label_classification_distillation":
                loss_fct = MyFocalLoss(version=4)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), outputs2.logits.view(-1, self.num_labels))
            elif self.config.problem_type == "single_label_classification_focalloss":
                # for focal loss
                loss_fct = FocalLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                
        # Combine losses if both are present
        if loss is not None and contrastive_loss is not None:
            loss += contrastive_loss * self.contrastive_learning_weight
        elif contrastive_loss is not None:
            loss = contrastive_loss        
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def compute_contrastive_loss(self, pooled_output_1: torch.Tensor, pooled_output_2: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Computes contrastive loss for SimCSE using two different dropout views.
        """
        batch_size = pooled_output_1.size(0)

        # Normalize the pooled outputs for contrastive loss
        pooled_output_1 = F.normalize(pooled_output_1, p=2, dim=1)
        pooled_output_2 = F.normalize(pooled_output_2, p=2, dim=1)

        # Compute similarity matrix between the two views
        similarity_matrix = torch.matmul(pooled_output_1, pooled_output_2.T) / temperature

        # Create labels for contrastive learning
        labels = torch.arange(batch_size).to(pooled_output_1.device)

        # Compute cross-entropy loss
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(similarity_matrix, labels)
        return loss