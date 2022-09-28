import torch
import random

from model.gadgets.my_metrics import Accuracy, VQAScore, Scalar

from model.modules.objectives import compute_vrar_recall

def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k=="mae_audio" or k=="mae_video":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k=="mosei":
                setattr(pl_module, f"{split}_{k}_accuracy2", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k=="moseiemo":
                setattr(pl_module, f"{split}_{k}_angry", Accuracy())
                setattr(pl_module, f"{split}_{k}_disgust", Accuracy())
                setattr(pl_module, f"{split}_{k}_fear", Accuracy())
                setattr(pl_module, f"{split}_{k}_happy", Accuracy())
                setattr(pl_module, f"{split}_{k}_sad", Accuracy())
                setattr(pl_module, f"{split}_{k}_surprise", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                
def epoch_wrapup(pl_module):
    torch.distributed.barrier()
    phase = "train" if pl_module.training else "val"
    the_metric = 0
    print("")
    print("=================================================")

    if pl_module.hparams.config["get_va_recall_metric"] and not pl_module.training:
        (vr_r1, vr_r5, vr_r10, ar_r1, ar_r5, ar_r10) = compute_vrar_recall(pl_module)
        print((vr_r1, vr_r5, vr_r10, ar_r1, ar_r5, ar_r10), pl_module.global_step)
        pl_module.logger.experiment.add_scalar(
            "recalls/vr_r1", vr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/vr_r5", vr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/vr_r10", vr_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ar_r1", ar_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ar_r5", ar_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ar_r10", ar_r10, pl_module.global_step
        )
        the_metric += vr_r1.item() + ar_r1.item()
        
    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        value = 0        
        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            print(f"{loss_name}/{phase}/score_epoch", value)
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)            
            
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            print(f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute())
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()            
            
                        
        elif loss_name == "vam" or loss_name == "vtm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            print(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
                        
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            print(f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
        elif loss_name == "mlm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            print(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
                        
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            print(f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
        elif loss_name == "mae_audio" or loss_name == "mae_video":
            value = - getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            print(f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
        elif loss_name == "mosei":
            value2 = getattr(pl_module, f"{phase}_{loss_name}_accuracy2").compute()
            
            pl_module.log(f"{loss_name}/{phase}/accuracy2_epoch", value2)
            print(f"{loss_name}/{phase}/accuracy2_epoch", value2)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy2").reset()
                        
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            print(f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
            the_metric += value2
            
        elif loss_name == "moseiemo":
                
            happy = getattr(pl_module, f"{phase}_{loss_name}_happy").compute()
            sad = getattr(pl_module, f"{phase}_{loss_name}_sad").compute()
            angry = getattr(pl_module, f"{phase}_{loss_name}_angry").compute()
            fear = getattr(pl_module, f"{phase}_{loss_name}_fear").compute()
            disgust = getattr(pl_module, f"{phase}_{loss_name}_disgust").compute()
            surprise = getattr(pl_module, f"{phase}_{loss_name}_surprise").compute()
            
            pl_module.log(f"{loss_name}/{phase}/happy_epoch", happy)
            pl_module.log(f"{loss_name}/{phase}/sad_epoch", sad)
            pl_module.log(f"{loss_name}/{phase}/angry_epoch", angry)
            pl_module.log(f"{loss_name}/{phase}/fear_epoch", fear)
            pl_module.log(f"{loss_name}/{phase}/disgust_epoch", disgust)            
            pl_module.log(f"{loss_name}/{phase}/surprise_epoch", surprise)
            
            print(f"{loss_name}/{phase}/happy_epoch", happy)
            print(f"{loss_name}/{phase}/sad_epoch", sad)
            print(f"{loss_name}/{phase}/angry_epoch", angry)
            print(f"{loss_name}/{phase}/fear_epoch", fear)
            print(f"{loss_name}/{phase}/disgust_epoch", disgust)            
            print(f"{loss_name}/{phase}/surprise_epoch", surprise)
            
            getattr(pl_module, f"{phase}_{loss_name}_happy").reset()
            getattr(pl_module, f"{phase}_{loss_name}_sad").reset()
            getattr(pl_module, f"{phase}_{loss_name}_angry").reset()
            getattr(pl_module, f"{phase}_{loss_name}_fear").reset()
            getattr(pl_module, f"{phase}_{loss_name}_disgust").reset()            
            getattr(pl_module, f"{phase}_{loss_name}_surprise").reset()
            
                        
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            print(f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
            the_metric += angry
            the_metric += disgust
            the_metric += fear
            the_metric += happy
            the_metric += sad
            the_metric += surprise
            
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            print(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
                        
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            print(f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)
    print("=================================================")
    torch.distributed.barrier()
    
    
def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return

