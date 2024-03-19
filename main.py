import cv2
import os.path

import EmbeddingAndExtraction as EaE

import metrics

watermark_address = "watermark.png"
watermark = EaE.fromWatermarkToMessage(watermark_address)
dk = "keykeykeykeykey"
ek = "key_rand"
folder = "CoverImages"
base = 2

if os.path.exists(folder) and os.path.isdir(folder):
    CoverImages = os.listdir(folder)
else:
    CoverImages = []

BPP = 0
PSNR = 0
BER = 0
SSIM = 0

for image_name in CoverImages:
    image_path = folder + "/" + image_name
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    new_image, clm, extract_keys = EaE.data_embedding(image, watermark, base, dk, ek)
    save1 = cv2.imwrite("StegoImages/Stego" + os.path.basename(image_path), new_image)

    recoveredImage, extractedWatermark = EaE.data_extraction(new_image, clm, extract_keys[0], extract_keys[1], base)
    save2 = cv2.imwrite("RecoveredImages/Recovered" + os.path.basename(image_path), recoveredImage)
    save3 = cv2.imwrite("ExtractedWatermarks/WatermarkExtractedFrom" + os.path.basename(image_path),
                        EaE.fromMessageToArray(extractedWatermark, (256, 256)))

    print("Метрики ", os.path.basename(image_path), ":")
    BPP += metrics.bpp(len(watermark), new_image.shape)
    print("BPP: ", str(metrics.bpp(len(watermark), new_image.shape)))
    PSNR += metrics.psnr(image, new_image, image.shape)
    print("PSNR: ", str(metrics.psnr(image, new_image, image.shape)))
    BER += metrics.ber(watermark, extractedWatermark, base)
    print("BER: ", str(metrics.ber(watermark, extractedWatermark, base)))
    SSIM += metrics.SSIM(image, recoveredImage)
    print("SSIM: ", str(metrics.SSIM(image, recoveredImage)))
    metrics.authenticationAnalysis(EaE.fromMessageToArray(extractedWatermark, (256, 256)),
                                   EaE.fromMessageToArray(watermark, (256, 256)), base)
    save4 = cv2.imwrite("ReversibilityAnalysis/Difference" + os.path.basename(image_path),
                        metrics.reversibilityAnalysis(new_image, image))
    print("\n")
print("Среднее BPP: ", str(BPP/len(CoverImages)))
print("Среднее PSNR: ", str(PSNR/len(CoverImages)))
print("Среднее BER: ", str(BER/len(CoverImages)))
print("Среднее SSIM: ", str(SSIM/len(CoverImages)))
