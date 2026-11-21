# Task 4 handoff (Thawat)

Task 4 คือจำแนกภาพเกม 5 คลาส (7-11k, fiveN, freefried, pube, roblock) วัดด้วย macro F1

## สถานะตอนนี้

- **Task 4 จบแล้ว** คอลัมน์ task4 ในไฟล์ส่งที่ดีที่สุด `kaggle_t12345_081771.csv` (คะแนนรวม 0.81771) มาจากโมเดลของเพื่อน `resnet18_cosine18` (ResNet18 pretrained, 18 epoch, cosine schedule, warmup 2 epoch) ทายด้วย TTA 5 view (หมุน ±8°, ย่อขยาย 0.95/1.05) ได้ 5-fold CV 0.97551
- งาน v3 ในโฟลเดอร์นี้ใช้สูตรเก่า 7 epoch ได้ประมาณ 0.95 **ไม่ได้ส่ง Kaggle** เพราะตามหลังโมเดลเพื่อนราว 0.03
- ลองรัน checkpoint ของเพื่อนบนเครื่องนี้ใหม่แล้ว ได้คำตอบตรงกับคอลัมน์ task4 ในไฟล์ 0.81771 อยู่ 25,883 จาก 25,889 แถว (99.98%) ยืนยันได้ว่าคอลัมน์นั้นมาจากโมเดลนี้จริง
- `final-notebook/task4.ipynb` ยังว่าง ถ้าจะใส่ ให้อิง notebook `Task4_gpu_followup.ipynb` ของเพื่อน ไม่ใช่ v3 ของฉัน

## มีอะไรในโฟลเดอร์นี้

| path | คืออะไร |
| --- | --- |
| `docs/task4-report.th.md`, `.docx` | รายงาน Task 4 ทีละขั้น ตั้งแต่ EDA จนถึงโมเดลสุดท้าย (ฉบับไทยเป็นหลัก) |
| `docs/task4-report.en.md`, `.docx` | ฉบับภาษาอังกฤษ เนื้อหาเดียวกัน |
| `eda/build_eda_near_duplicate_v3.ipynb` | หาภาพเกือบซ้ำด้วย pHash/dHash แล้วบันทึก dataset `eda_near_duplicate_v3` |
| `data/eda/eda_near_duplicate_v3*` | dataset ที่ notebook ข้างบนสร้าง (card JSON + parquet) |
| `models/build_model_v3.ipynb` | เทรน ResNet18 สูตรเก่าใหม่บน fold 1-2 ที่แยกกลุ่มภาพเกือบซ้ำ, วัดผลรั่ว, ทดลอง TTA, ดู error |
| `models/build_model_v3_full_tta.ipynb` | เทรนบน train ทั้งหมด แล้วสร้าง CSV ทั้งแบบปกติและแบบ TTA |
| `models/tta.py` | ฟังก์ชัน TTA ที่สอง notebook ข้างบนเรียกใช้ |
| `save/*v3*` | card ของโมเดล, ผล evaluation, ตัวรวม TTA (`.joblib`) |
| `submission/t4_resnet18_geometry_full_v3*.csv` | คำตอบ test ของ v3 (ไม่ได้ส่ง) |
| `scripts/repro_friend_t4.py` | รัน checkpoint ของเพื่อนพร้อม TTA แล้วเทียบกับไฟล์ 0.81771 |
| `scripts/md_to_docx.py` | แปลงรายงาน .md เป็น .docx |

## จะรัน notebook ต้องวางกลับเข้า repo เดิม

notebook ทั้งหมดเรียก `src.pipeline` และอ่าน artifact ตาม path ของ repo `cpe342-ml-kaggle` จึงรันจากโฟลเดอร์นี้ตรงๆ ไม่ได้ ให้ copy กลับไปตามนี้

```
eda/*          -> cpe342-ml-kaggle/src/task4/eda/
models/*       -> cpe342-ml-kaggle/src/task4/models/
data/eda/*     -> cpe342-ml-kaggle/src/task4/data/eda/
save/*         -> cpe342-ml-kaggle/src/task4/save/
submission/*   -> cpe342-ml-kaggle/src/task4/submission/
docs/*         -> cpe342-ml-kaggle/docs/task4-report/
```

notebook model ทั้งสองอ่าน dataset `eda_near_duplicate_v3` ที่อัปไว้แล้วใน `data/eda/` และต้องมีภาพ raw ของ Task 4 ใน `src/task4/data/raw/` ด้วย

รันสคริปต์:

```
python scripts/repro_friend_t4.py <โฟลเดอร์ผลลัพธ์> <โฟลเดอร์ Kaggle-Final>
python scripts/md_to_docx.py docs/task4-report.th.md docs/task4-report.th.docx <โฟลเดอร์ที่ลง python-docx> th
```

`<โฟลเดอร์ Kaggle-Final>` ต้องมี `cpe342-ml-kaggle/`, `of_friends/resnet18_cosine18_pretrained.pt` และ `kaggle_t12345_081771.csv` อยู่ข้างใน

## ผลที่ได้จากงาน v3 (สูตรเก่า 7 epoch)

- รัน fold 1 ของ `build_model_v2` ซ้ำได้ 0.95246 เท่าเดิมทุกหลัก
- ภาพเกือบซ้ำ (pHash ≤ 4, dHash ≤ 16) มี 1,806 ภาพที่อยู่ในกลุ่มหลายภาพ ส่วนมากเป็น 7-11k ผลรั่วข้าม fold ดันคะแนนขึ้นแค่ประมาณ 0.00076 จึงไม่ใช่ปัญหาใหญ่
- TTA ทั้ง 5 view: fold 1 +0.0031, fold 2 +0.0018
- error ส่วนใหญ่ถูกทายผิดเป็น fiveN ภาพที่มืดสุด 25% ทายถูก 0.938 เทียบกับไม่ต่ำกว่า 0.964 ในกลุ่มอื่น ความเบลอไม่ใช่ปัญหา

## ของที่ไม่ได้อัป (อยู่บนเครื่องฉัน)

- checkpoint `.pt` 3 ไฟล์ ไฟล์ละ 45 MB (`t4_resnet18_geometry_fold1_v3`, `fold2_v3`, `full_v3`) ไม่ได้ใช้กับไฟล์ส่ง เทรนใหม่จาก notebook ได้
- cache `.npz` ของ probability แต่ละ view ใน `src/task4/models/.cache/`
- ภาพ raw ของ Task 4 (โหลดจากลิงก์ Google Drive ใน README ของ zip Kaggle)
- `src/task4/save/evaluation_index.json` ในเครื่องฉันมี entry ของ v3 เพิ่มเข้าไป ถ้าวาง `save/*` กลับแล้วรัน `evaluate_model` ใหม่ index จะอัปเดตเอง

## ถ้าจะทำต่อ

1. EfficientNet-B0 ใน notebook เพื่อนรันได้ fold 3 = 0.97930 และ fold 4 = 0.97955 (แบบ TTA) แล้วหยุดกลาง fold 5 ถ้ารันครบ 5 fold แล้วชนะ cosine18 ก็ลอง ensemble สองตัว
2. เพิ่ม augmentation ปรับความสว่าง/contrast เพราะ test มืดกว่า train (brightness เฉลี่ย 0.414 เทียบกับ 0.451) และภาพมืดผิดบ่อยกว่า
3. ภาพเกือบซ้ำ 4 คู่ที่ notebook เพื่อนเจอ ยังไม่มีใครเปิดดูด้วยตา

## Environment ที่ใช้

Python 3.12, torch 2.11 + CUDA 12.8, RTX 3050 6 GB เทรน fold เดียวสูตร 7 epoch ใช้ประมาณ 27 นาที
zip ของ Kaggle ใช้ชื่อโฟลเดอร์ `01_binary_classification` แบบนั้น และภาพ Task 4 ต้องโหลดแยกจาก Google Drive ทำให้ `src/load-raw-dataset.py` ใช้ตรงๆ ไม่ได้
