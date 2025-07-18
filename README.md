# Style-Genie
It is an Intelligent model which predicts Skintone and recommends outfit based on skintone,gender and occation using Deep Learning and API.

• Built a deep learning model using ResNet50 with CLAHE and skin masking for accurate 
skin tone classification. 
• Integrated with a fashion API to recommend outfits based on skin tone, gender, and 
occasion. 

Steps:
1) Dataset: Monk Skintone Dataset (https://skintone.google/mste-dataset). Skintone is classified into 10 types i.e, from Very fair to Darkest.
2) Data Preprocessing: Removed some outliers, Mapped the names of images in .csv file to the images folder.
3) Image Processing: -> MTCNN(Multi-task Cascaded CNN) for face detection. Removes unwanted regions or noise in the image.
                     -> CLAHE(Contrast Limited Adaptive Histogram Equalization) to improve visibility of skin features without over-amplifying noise.
