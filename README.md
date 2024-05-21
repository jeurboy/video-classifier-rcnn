# จัด folder ตามนี้เลย

- dataset/train/normal/*.mp4
- dataset/train/shoplifter/*.mp4

- dataset/test/normal/*.mp4
- dataset/test/shoplifter/*.mp4


# รัน เพื่อเตรียม dataset
prepare_train_data.py จะได้ไฟล์ train.csv
prepare_test_data.py จะได้ไฟล์ test.csv

# เรื่มเทรน
รัน train.py เพื่อเทรนโมเดล จะได้ file model/model.keras ออกมา (สร้าง folder model รอไว้ ไม่งั้น error)

# รันเทส
รัน predict.py (ปัจจุบันเอาไฟล์มาจาก test.csv)
