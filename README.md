# Style-Genie
It is an Intelligent model which predicts Skintone and recommends outfit based on skintone,gender and occation using Deep Learning and API.

• Built a deep learning model using ResNet50 with CLAHE and skin masking for accurate 
skin tone classification. 

• Integrated with a fashion API to recommend outfits based on skin tone, gender, and 
occasion. 


Steps:
1) Dataset: Monk Skintone Dataset (https://skintone.google/mste-dataset). Skintone is classified into 10 types i.e, from Very fair to Darkest.
2) Data Preprocessing: Removed some outliers, Mapped the names of images in .csv file to the images folder.
3) Image Processing:
   
                     -> MTCNN(Multi-task Cascaded CNN) for face detection. Removes unwanted regions or noise in the image.
                     -> CLAHE(Contrast Limited Adaptive Histogram Equalization) to improve visibility of skin features without over-amplifying noise.
                     -> HSV-Based Skin Masking used to highlight only skin regions.
4) Image Conversion: Image resizing and RGB conversion.
5) Feature extraction with ResNet50 - we no need to train low level features. We customized Classification Head, replaced original classification with tailored number of specific skintone classes.
   | *Stage*               | *Technique*                  | *Impact*                                |
| ----------------------- | ------------------------------ | ----------------------------------------- |
| Contrast Enhancement    | CLAHE                          | Improves clarity in low/high light images |
| Background Removal      | YCrCb Masking                  | Focus on skin only                        |
| Deep Feature Extraction | ResNet50 (transfer learning)   | Robust pretrained feature space           |
| Classifier              | Dense + Dropout + L2           | Learns skin-tone specific features        |
| Generalization          | Data Augmentation + Callbacks  | Prevents overfitting                      |
| Fine-tuning             | Unfreeze last 50 ResNet layers | Task-specific feature adjustment          |
| Output Layer            | Softmax                        | Multiclass prediction                     |
| Performance Tracking    | EarlyStopping + LR Scheduler   | Efficient training process                |
