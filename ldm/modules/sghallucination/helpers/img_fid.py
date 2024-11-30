
from os.path import join as opj
import os
import json
from shutil import copyfile


def v0():
    pred_img_dir = '../GLIGEN/generation_samples'
    model_names = ['chat_icl', 'layoutdm', 'vqdiffusion', 'blt', 'maskgit', 'layouttrans']

    json_names_gt = ['only_numeral.json', 'sample_only_spatial.json', 'sample_only_semantic.json',
                     'mix_relation.json', 'sample_non_relation.json']


    for model_name in model_names:
        for j_name in json_names_gt:
            print(f'{model_name} + {j_name}...')
            category = j_name.split('.')[0]
            category = category.split('_')[-2:]
            category = '_'.join(category)

            pred_path = pred_img_dir + '/predict_' + category + '/' + model_name
            gt_path = pred_img_dir + '/predict_' + category + '/gt'
            os.system(f'python -m pytorch_fid {pred_path} {gt_path} --device cuda:1')

def v1():
    model_names = ['rl']
    # json_names_gt = ['only_numeral.json', 'sample_only_spatial.json', 'sample_only_semantic.json',
    #                     'mix_relation.json', 'sample_non_relation.json']

    category_list = ["only_numeral", "sample_only_spatial", "sample_only_semantic", "mix_relation", "sample_non_relation"]
    for category in category_list:
        # category = 'sample_only_semantic'  # sample_only_semantic
        print(category)
        gt_path = "/storage_fast/lgqu/generation/GLIGEN/generation_samples/gt/" + category
        # pred_path = './GLIGEN_v0/generation_samples/exp6_aesR_lr1e-3_t1_iou_itr_2023_04_30_15_52_38/' + category
        # pred_path = './GLIGEN_v0/generation_samples/exp6_aesR_lr1e-3_t1_iou_2023_04_30_08_07_48/' + category
        pred_path = "/storage/sqwu/diffusion/VQ-Diffusion/generation_samples/" + category

        for model_name in model_names:
            print(f'{model_name} ...')
            os.system(f'python -m pytorch_fid {pred_path} {gt_path} --device cuda:1')

gt_path = "/storage_fast/lgqu/generation/GLIGEN/generation_samples/gt/"
pred_path = "/storage/sqwu/diffusion/VQ-Diffusion/generation_samples/"
os.system(f'python -m pytorch_fid {pred_path} {gt_path} --device cuda:1')


# v1()

