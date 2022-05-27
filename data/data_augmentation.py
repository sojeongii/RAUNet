import os
import cv2
root_path = "./cv_project"
train_dir = os.path.join(root_path, "train")
save_dir = os.path.join(root_path, "aug")
examples = [os.path.join(root_path, "train", dirs) for dirs in os.listdir(train_dir)]

print(len(examples))
for f in examples:
    print(f)
    file = cv2.imread(f)
    file = cv2.cvtColor(file , cv2.COLOR_BGR2RGB)
    h_file = cv2.flip(file,1)
    v_file = cv2.flip(file, 0)
    r_file = cv2.rotate(file, cv2.ROTATE_180)
    cv2.imwrite(os.path.join(train_dir, 'h_flip_image{}.png'.format(examples.index(f))), h_file)
    cv2.imwrite(os.path.join(train_dir, 'v_flip_image{}.png'.format(examples.index(f))), v_file)
    cv2.imwrite(os.path.join(train_dir, 'r_flip_image{}.png'.format(examples.index(f))), r_file)