import cv2
import detectron_infer

im = cv2.imread("./image1163.jpg")
img, result_info = detectron_infer.get_detectron_result(im)
cv2.imwrite("result.jpg", img)
print("---------------------->", result_info)

