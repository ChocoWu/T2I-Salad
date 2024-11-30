import json
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import os
from PIL import Image
import torch.nn.functional as F
from shutil import copyfile
import torch
import numpy as np
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gt_img_dir = "/storage_fast/lgqu/generation/dataset/coco/images/val2014"
clip_version = 'openai/clip-vit-large-patch14' # openai/clip-vit-base-patch32
tokenizer = AutoTokenizer.from_pretrained(clip_version)
model = CLIPModel.from_pretrained(clip_version).to('cuda')
processor = AutoProcessor.from_pretrained(clip_version)
model.eval()
# captions = []
# imgs = []
# gt_imgs = []



# model_name = 'chat_icl'
# model_names = ['chat_icl', 'layoutdm', 'vqdiffusion', 'blt', 'maskgit', 'layouttrans']
model_names = ['rl']

# exp='exp6_aesR_lr1e-3_t1_iou_itr_2023_04_28_16_29_03'
# exp='exp6_aesR_lr1e-3_t1_iou_2023_04_27_16_28_53'
# exp = 'random_shot4'
# exp = 'exp6_aesR_lr1e-3_t1_iou_2023_04_30_08_07_48'
# exp = 'exp6_aesR_lr1e-3_t1_iou_itr_2023_04_30_15_52_38'
exp = 'random_shot4_v1'

pred_img_dir = "/storage/sqwu/diffusion/VQ-Diffusion/generation_samples"
# pred_img_dir = './GLIGEN_v0/generation_samples/' + exp
# pred_img_dir = '/storage_fast/lgqu/generation/GLIGEN/generation_samples/layouttrans'
data_dir_gt = '/storage/sqwu/diffusion/layout-dm/download/datasets/coco2014-max25'
# j_name = 'five_category.json' # five_category
# json_names_gt = ['sample_only_spatial.json']
json_names_gt = ['only_numeral.json', 'sample_only_spatial.json', 'sample_only_semantic.json',
                 'mix_relation.json', 'sample_non_relation.json']

# only_numeral -- [:155], sample_only_spatial -- [155: 355], sample_only_semantic -- [355: 555]
    # mix_relation -- [555: 743], sample_non_relation -- [743: ]

# with open(os.path.join(data_dir_gt, j_name)) as f:
#     gt_dict = json.load(f)
#
# gt_dict = gt_dict[355: 555]
# print('#################### ' + str(len(gt_dict)) + ' ####################')

def category():
    for model_name in model_names:
        for j_name in json_names_gt:
            print(f'{model_name} + {j_name}...')
            category = j_name.split('.')[0]
            category = category.split('_')[-2:]
            category = '_'.join(category)

            with open(os.path.join(data_dir_gt, j_name)) as f:
                gt_dict = json.load(f)

            captions = []
            imgs = []
            gt_imgs = []

            for x in gt_dict:
                captions.append(x['captions'])
                name = f'{model_name}_' + x['name'].split('.')[0] + '.png'
                # name = 'layoutdm_' + x['name'].split('.')[0] + '.png'
                img = Image.open(os.path.join(pred_img_dir + '/predict_' + category, name))
                # print(img.size)
                imgs.append(img)
                gt_img = Image.open(os.path.join(gt_img_dir, x['name']))
                gt_img.resize((512, 512)).save(pred_img_dir + '/predict_' + category + '/gt/' + x['name'])
                gt_imgs.append(gt_img)
                # copyfile(os.path.join(gt_img_dir, x['name']), os.path.join(pred_img_dir + '/gt/', x['name']))


            with torch.no_grad():
                # # CLIP sim
                inputs = tokenizer(captions, padding=True, return_tensors="pt").to('cuda')
                text_features = model.get_text_features(**inputs)
                inputs = processor(images=imgs, return_tensors="pt").to('cuda')
                img_features = model.get_image_features(**inputs)
                inputs = processor(images=gt_imgs, return_tensors="pt").to('cuda')
                img_features_gt = model.get_image_features(**inputs)

                text_features = F.normalize(text_features, dim=-1)
                img_features = F.normalize(img_features, dim=-1)
                img_features_gt = F.normalize(img_features_gt, dim=-1)

                all_sims = (text_features * img_features).sum(dim=-1)
                print('CLIP_sim (t-i)', all_sims.mean().item())

                all_sims_i2i = (img_features_gt * img_features).sum(dim=-1)
                print('CLIP_sim (i-i)', all_sims_i2i.mean().item())


def all():
    for model_name in model_names:
        j_name = 'five_category.json'
        print(f'{model_name} + {j_name}...')
        category = j_name.split('.')[0]
        category = category.split('_')[-2:]
        category = '_'.join(category)

        with open(os.path.join(data_dir_gt, j_name)) as f:
            gt_dict = json.load(f)

        captions = []
        imgs = []
        gt_imgs = []
        for x in gt_dict:
            captions.append(x['captions'])
            name = x['name'].split('.')[0] + '.png'
            # name = 'layoutdm_' + x['name'].split('.')[0] + '.png'
            img = Image.open(os.path.join(pred_img_dir + '/' + model_name + '/' + name))
            img = img.convert('RGB')
            img.resize((512, 512))
            # print(img.size)
            imgs.append(np.array(img))
            gt_img = Image.open(os.path.join(gt_img_dir, x['name']))
            gt_img = gt_img.convert('RGB')
            gt_img.resize((512, 512))
            gt_imgs.append(np.array(gt_img))
            # copyfile(os.path.join(gt_img_dir, x['name']), os.path.join(pred_img_dir + '/gt/', x['name']))

        # imgs = np.stack(imgs, axis=0)
        # gt_imgs = np.stack(gt_imgs, axis=0)
        with torch.no_grad():
            # # CLIP sim
            inputs = tokenizer(captions, padding=True, return_tensors="pt").to('cuda')
            text_features = model.get_text_features(**inputs)
            inputs = processor(images=imgs, return_tensors="pt").to('cuda')
            img_features = model.get_image_features(**inputs)
            inputs = processor(images=gt_imgs, return_tensors="pt").to('cuda')
            img_features_gt = model.get_image_features(**inputs)

            text_features = F.normalize(text_features, dim=-1)
            img_features = F.normalize(img_features, dim=-1)
            img_features_gt = F.normalize(img_features_gt, dim=-1)

            all_sims = (text_features * img_features).sum(dim=-1)
            print('CLIP_sim (t-i)', all_sims.mean().item())

            all_sims_i2i = (img_features_gt * img_features).sum(dim=-1)
            print('CLIP_sim (i-i)', all_sims_i2i.mean().item())

def all_v1(gt_dict):
    for model_name in model_names:
        # pred_img_dir = '/storage_fast/lgqu/generation/GLIGEN/generation_samples/' + model_name
        # j_name = 'five_category.json'
        print(f'{model_name} + {j_name} - {len(gt_dict)}...')
        category = j_name.split('.')[0]
        category = category.split('_')[-2:]
        category = '_'.join(category)


        captions = []
        imgs = []
        gt_imgs = []
        for x in tqdm(gt_dict):
            captions.append(x['captions'])
            pred_name = x['name'].split('.')[0] + '.png'
            # name = 'layoutdm_' + x['name'].split('.')[0] + '.png'
            img = Image.open(os.path.join(pred_img_dir, pred_name))
            img = img.convert('RGB')
            img.resize((512, 512))
            # print(img.size)
            imgs.append(np.array(img))
            gt_img = Image.open(os.path.join(gt_img_dir, x['name']))
            gt_img = gt_img.convert('RGB')
            gt_img.resize((512, 512))
            gt_imgs.append(np.array(gt_img))
            # copyfile(os.path.join(gt_img_dir, x['name']), os.path.join(pred_img_dir + '/gt/', x['name']))

        # imgs = np.stack(imgs, axis=0)
        # gt_imgs = np.stack(gt_imgs, axis=0)
        with torch.no_grad():
            # # CLIP sim
            inputs = tokenizer(captions, padding=True, return_tensors="pt").to('cuda')
            text_features = model.get_text_features(**inputs)
            inputs = processor(images=imgs, return_tensors="pt").to('cuda')
            img_features = model.get_image_features(**inputs)
            inputs = processor(images=gt_imgs, return_tensors="pt").to('cuda')
            img_features_gt = model.get_image_features(**inputs)

            text_features = F.normalize(text_features, dim=-1)
            img_features = F.normalize(img_features, dim=-1)
            img_features_gt = F.normalize(img_features_gt, dim=-1)

            all_sims = (text_features * img_features).sum(dim=-1)
            print('CLIP_sim (t-i)', all_sims.mean().item())

            all_sims_i2i = (img_features_gt * img_features).sum(dim=-1)
            print('CLIP_sim (i-i)', all_sims_i2i.mean().item())


def all_v2():
    # calculate the similarity between captions and images generated by VQDiffusion
    model_name = "VQDiffusion"
    
    captions = []
    imgs = []
    gt_imgs = []
    for j_name in json_names_gt:
        if os.path.exists(os.path.join(data_dir_gt, j_name)):
            with open(os.path.join(data_dir_gt, j_name)) as f:
                gt_dict = json.load(f)
        else:
            print(f"{j_name} is not exist")
            continue


        print(f'{model_name} + {j_name} - {len(gt_dict)}...')
        category = j_name.split('.')[0]
        # category = category.split('_')[-2:]
        # category = '_'.join(category)

        for x in tqdm(gt_dict):
            captions.append(x['captions'])
            pred_name = x['name']
            # name = 'layoutdm_' + x['name'].split('.')[0] + '.png'
            img = Image.open(os.path.join(pred_img_dir, category, pred_name))
            img = img.convert('RGB')
            img.resize((512, 512))
            # print(img.size)
            imgs.append(np.array(img))
            gt_img = Image.open(os.path.join(gt_img_dir, x['name']))
            gt_img = gt_img.convert('RGB')
            gt_img.resize((512, 512))
            gt_imgs.append(np.array(gt_img))
            # copyfile(os.path.join(gt_img_dir, x['name']), os.path.join(pred_img_dir + '/gt/', x['name']))

        # imgs = np.stack(imgs, axis=0)
        # gt_imgs = np.stack(gt_imgs, axis=0)
    with torch.no_grad():
        # # CLIP sim
        inputs = tokenizer(captions, padding=True, return_tensors="pt").to('cuda')
        text_features = model.get_text_features(**inputs)
        inputs = processor(images=imgs, return_tensors="pt").to('cuda')
        img_features = model.get_image_features(**inputs)
        inputs = processor(images=gt_imgs, return_tensors="pt").to('cuda')
        img_features_gt = model.get_image_features(**inputs)

        text_features = F.normalize(text_features, dim=-1)
        img_features = F.normalize(img_features, dim=-1)
        img_features_gt = F.normalize(img_features_gt, dim=-1)

        all_sims = (text_features * img_features).sum(dim=-1)
        print('CLIP_sim (t-i)', all_sims.mean().item())

        all_sims_i2i = (img_features_gt * img_features).sum(dim=-1)
        print('CLIP_sim (i-i)', all_sims_i2i.mean().item())

# category()
all_v2()
