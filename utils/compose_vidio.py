import os
import mmcv
import numpy as np
import cv2
import mmcv
import json
PATH = './visuals'
video_path = 'seg_det_demo_best_v1.mp4'

if __name__ == '__main__':
    count = 0
    with open('./data.json', 'r') as fp:
        data = json.load(fp)
    bevformer_results = mmcv.load(
        './test/segdet_front_finetune/Sat_May_27_02_00_45_2023/pts_bbox/results_nusc.json')
    sample_token_list = list(bevformer_results['results'].keys())[1000:2000]
    for id in range(0, 100):

        if sample_token_list[id] + '.png' not in os.listdir(PATH):
            continue
            
        print(f"handle {PATH + sample_token_list[id] +'.png'}")
        count += 1
        im = os.path.join(PATH, sample_token_list[id] + '.png')
        im = cv2.imread(im) # prediction

        # add gt
        gt = os.path.join(PATH, sample_token_list[id] + '_gt.png')
        gt = cv2.imread(gt) # prediction

        # add input
        inp = data[sample_token_list[id]]
        inp = cv2.imread(inp)
        inp = cv2.resize(inp, (720, 400), interpolation=cv2.INTER_CUBIC)
        
        whole = np.zeros((400,1160,3))
        whole[:,:720,:] = inp
        whole[:,740:940,:] = np.rot90(im)
        whole[:,960:,:] = np.rot90(gt)
        whole = whole.astype(np.uint8)

        if count == 1:
            fps, w, h = 5, whole.shape[1], whole.shape[0]
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
            
        out.write(whole)
    out.release()
    print('Done!')