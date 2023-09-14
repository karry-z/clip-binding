import os
import shutil

did = 0
src_dir = '/user/work/pu22650/data/rel/images/train'
dst_dir = '/user/home/pu22650/work/clip-binding-out/DM_ref'
for sid in range(10000):
    src_img_name = f'CLEVR_train_rel_{sid:06}.png'
    src_img_path = os.path.join(src_dir, src_img_name)
    for _ in range(4):
        dst_img_name = f'CLEVR_train_rel_{did:06}.png'
        dst_img_path = os.path.join(dst_dir, dst_img_name)
        did += 1
        shutil.copy(src_img_path, dst_img_path)