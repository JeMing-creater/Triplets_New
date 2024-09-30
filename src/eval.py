import ivtmetrics

def val(config, model, dataloader, activation, step=0, train=False):
    model.eval()
    rec = ivtmetrics.Recognition(config.dataset.class_num)
    rec.reset_global()
    
    if train == False:
        data_set = 'Val'
    else:
        data_set = 'Train'
    
    for _, (img, (_, _, _, y)) in enumerate(dataloader):
        _, _, _, triplet = model(img)
        logit_ivt  = triplet  
        preds = activation(logit_ivt).detach().cpu()

        rec.update(y.float().detach().cpu(), preds)
        step += 1
    
    rec.video_end()
    
    # compute the final mAP for all the test videos
    imAP   = rec.compute_video_AP('i')['mAP']
    vmAP   = rec.compute_video_AP('v')['mAP']
    tmAP   = rec.compute_video_AP('t')['mAP']
    ivmAP  = rec.compute_video_AP('iv')['mAP']
    itmAP  = rec.compute_video_AP('it')['mAP']
    ivtmAP = rec.compute_video_AP('ivt')['mAP']
    # ivt_ap = rec.compute_video_AP('ivt')['AP']
    
    itopk   = rec.topK(config.trainer.top, 'i')
    ttopk   = rec.topK(config.trainer.top, 't')
    vtopk   = rec.topK(config.trainer.top, 'v')
    ivttopk = rec.topK(config.trainer.top, 'ivt')
    
    metrics = {
        f'{data_set}/I': round(imAP , 3),
        f'{data_set}/V': round(vmAP , 3),
        f'{data_set}/T': round(tmAP , 3),
        f'{data_set}/IV': round(ivmAP , 3),
        f'{data_set}/IVM': round(vmAP , 3),
        f'{data_set}/ITM': round(itmAP , 3),
        f'{data_set}/IVT': round(ivtmAP , 3),
        f'{data_set}/i-topk': itopk ,
        f'{data_set}/t-topk': ttopk ,
        f'{data_set}/v-topk': vtopk ,
        f'{data_set}/ivt-topk': ivttopk 
    }
    
    return metrics, step