# รายงาน Task 4: Game Title Detection

ฉบับภาษาไทยเป็นฉบับหลัก ฉบับภาษาอังกฤษอยู่ที่ [task4-report.en.md](task4-report.en.md) หัวข้อเลขเดียวกันทั้งสองฉบับ

## 1. โจทย์

ทายว่าภาพหน้าจอเกมขนาด 230×120 แต่ละรูปมาจากเกมไหน มี 5 คลาส คือ `7-11k`, `fiveN`, `freefried`, `pube`, `roblock` วัดผลด้วย macro F1 ซึ่งให้น้ำหนักทุกคลาสเท่ากัน Task 4 เป็น task เดียวที่อนุญาตให้ใช้ pretrained model

คะแนนในรายงานนี้เป็น macro F1 จาก validation ที่แบ่งเองทั้งหมด ไม่ใช่คะแนน Kaggle เพราะ Kaggle แสดงแค่คะแนนรวมของ 5 task

## 2. ขั้นตอนการทำงาน

### ขั้นที่ 1 สำรวจข้อมูล

Notebook: `src/task4/eda/build_eda_original.ipynb`

| คลาส | จำนวนรูป train | สัดส่วน |
| --- | ---: | ---: |
| 7-11k | 12,429 | 39.4% |
| pube | 5,926 | 18.8% |
| freefried | 5,596 | 17.7% |
| fiveN | 4,407 | 14.0% |
| roblock | 3,188 | 10.1% |

- train 31,546 รูป, test 25,889 รูป ทุกรูปขนาด 230×120 และไม่มีไฟล์ภาพหาย
- ไม่มีภาพซ้ำกันใน train (เทียบ SHA-256 ของไฟล์) และไม่มีภาพที่ซ้ำกันระหว่าง train กับ test
- ภาพ test มืดกว่า train เล็กน้อย ความสว่างเฉลี่ย 0.414 เทียบกับ 0.451
- คลาสไม่สมดุล 7-11k มีมากกว่า roblock เกือบ 4 เท่า แต่ macro F1 ให้น้ำหนักสองคลาสนี้เท่ากัน

บันทึก dataset แรก `eda_original_v1` เป็น pixel ย่อเหลือ 16×8 สี RGB (384 ค่าต่อรูป) เพื่อใช้เป็น baseline

### ขั้นที่ 2 สร้าง feature จากภาพ

Notebook: `src/task4/eda/build_eda_1.ipynb`

สกัด feature แบบดั้งเดิมได้ 1,228 ค่าต่อรูป:

| กลุ่ม feature | จำนวน | เก็บข้อมูลอะไร |
| --- | ---: | --- |
| pixel 16×8 RGB | 384 | ภาพย่อ |
| mean/std ของ R, G, B | 6 | สีรวมทั้งภาพ |
| ค่าเฉลี่ยสีแบ่ง 2×4 ช่อง | 24 | สีตามตำแหน่งบนจอ |
| histogram ของ H, S, V | 48 | การกระจายสี |
| HOG | 756 | ทิศทางของขอบ |
| LBP | 10 | ลวดลาย/texture |

ทุก feature ไม่มีค่าว่างและไม่มีคอลัมน์ที่ค่าคงที่ สีเฉลี่ยของแต่ละคลาสใกล้กันเกินกว่าจะแยกเกมได้ด้วยสีอย่างเดียว

### ขั้นที่ 3 โมเดลรอบแรก (machine learning แบบดั้งเดิม)

Notebook: `src/task4/models/build_model.ipynb`

แบ่ง validation 20% แบบ stratified และให้ภาพที่ hash เดียวกันอยู่ฝั่งเดียวกัน แล้วเทียบ 7 โมเดล (LightGBM ลอง 2 ขนาดคือ 15 และ 31 leaves) แต่ละตัวลองทั้งบน pixel และบน feature จากขั้นที่ 2 จากนั้นจูน hyperparameter ด้วย 3 fold บนรูป 9,000 รูป

| โมเดล | pixel | feature |
| --- | ---: | ---: |
| Majority (ทายคลาสใหญ่สุดเสมอ) | 0.11305 | |
| Logistic regression | 0.41192 | 0.72128 |
| Linear SVM (SGD) | 0.33276 | 0.67859 |
| ExtraTrees | 0.52570 | 0.61219 |
| HistGradientBoosting (จูนแล้ว) | 0.57489 | 0.75287 |
| LightGBM 15 leaves (จูนแล้ว) | 0.54642 | 0.74278 |
| **LightGBM 31 leaves (จูนแล้ว)** | 0.58965 | **0.77164** |

- feature จากขั้นที่ 2 ดีกว่า pixel ทุกโมเดล
- ดีที่สุดแค่ 0.77 และ roblock ได้ F1 ประมาณ 0.68 ยังผิดเยอะ

### ขั้นที่ 4 test.csv ถูกเปลี่ยน

Notebook: `src/task4/eda/build_eda_original_v2.ipynb`, `build_eda_1_v2.ipynb`

อาจารย์เปลี่ยนเฉพาะ `test.csv` ส่วนภาพและ label ของ train เหมือนเดิมทุก byte จึงตรวจ hash ของภาพแล้วผูก feature เดิมเข้ากับ ID ใหม่ (`eda_original_v2`, `eda_1_v2`) ไม่ต้องสกัด feature ใหม่

### ขั้นที่ 5 เปลี่ยนมาใช้ pretrained ResNet18

Notebook: `src/task4/models/build_model_v2.ipynb`

ทุกรูปถูกย่อเป็น 112×224 ซึ่งคงสัดส่วนเดิม เติมขอบดำบนล่างให้เป็น 224×224 แล้ว normalize ด้วยค่า mean/std ของ ImageNet ทุกการทดลองเริ่มจาก weights เดียวกัน ใช้ AdamW lr 1e-4, batch 32, seed 42, loss ถ่วงน้ำหนักตามคลาส และเลือก epoch ที่ดีที่สุดจาก validation (หยุดเมื่อไม่ดีขึ้น 3 epoch)

ทดลองทีละอย่างบน fold เดียวกัน แล้วยืนยันบน fold ที่ 2

| วิธี | fold 1 | fold 2 |
| --- | ---: | ---: |
| LightGBM จากขั้นที่ 3 | 0.77164 | 0.77532 |
| ResNet18 ล็อก backbone เทรนแค่ชั้นสุดท้าย | 0.70728 | |
| Fine-tune ทั้งโมเดล ไม่มี augmentation | 0.94637 | 0.94962 |
| **Fine-tune + geometry augmentation** | **0.95246** | **0.95528** |
| Fine-tune + geometry + blur/noise | 0.95533 | 0.94346 |
| Fine-tune + หมุน ±180° | 0.94936 | |

geometry augmentation คือหมุนสุ่มได้ถึง ±50°, บิด perspective และย่อขยาย 0.8–1.18 เท่า ใช้กับภาพ train เท่านั้น

- การ fine-tune ทั้งโมเดลเป็นตัวที่ทำให้คะแนนกระโดดจาก 0.77 เป็น 0.95
- geometry ดีกว่าไม่มี augmentation ทั้งสอง fold (+0.006 ทั้งคู่)
- blur/noise ดีขึ้นบน fold 1 แต่แย่ลงมากบน fold 2 จึงไม่เลือก
- รัน geometry บน fold 3 เพื่อยืนยันได้ 0.95568 เฉลี่ย 3 fold คือ **0.95447** F1 ราย class บน fold 3: 7-11k 0.975, fiveN 0.952, freefried 0.969, pube 0.962, roblock 0.919

### ขั้นที่ 6 เทรนบน train ทั้งหมดและส่งครั้งแรก

เทรนสูตร geometry บน train ทั้ง 31,546 รูป 7 epoch (ค่ากลางของ epoch ที่ดีที่สุดใน 3 fold คือ 7, 8, 7) ได้โมเดล `t4_resnet18_geometry_full_v2` แล้วสร้าง `src/task4/submission/t4_resnet18_geometry_full_v2.csv`

### ขั้นที่ 7 Follow-up: ตั้งการทดลองใหม่ให้เทียบกันได้ยุติธรรม

Notebook: `Kaggle-Final/of_friends/Task4_gpu_followup.ipynb` (นอก repo รันเดี่ยวได้บน Colab/Kaggle GPU)

ขั้นที่ 5 เลือก epoch จาก validation และเลือกสูตรจาก fold เดียวกับที่ใช้รายงานคะแนน รอบนี้จึงเปลี่ยนกติกา:

- กำหนดจำนวน epoch ของทุกสูตรไว้ก่อนเทรน แล้วใช้ epoch สุดท้ายเสมอ validation ไม่ได้เลือก checkpoint
- เทรนสูตรเดิม (7 epoch) ใหม่บน fold ชุดเดียวกันเป็น baseline เพื่อให้เทียบกันได้ตรงๆ
- ตั้งกติกาเลือกสูตรไว้ก่อน: ต้องดีกว่า baseline เฉลี่ยอย่างน้อย 0.001 และดีกว่าอย่างน้อย 4 จาก 5 fold
- น้ำหนักของแต่ละคลาสใน loss คำนวณจาก train ของ fold นั้นเอง

ก่อนแบ่ง fold ตรวจภาพซ้ำอีกรอบ ภาพคู่ที่จะนับว่าเกือบซ้ำต้องผ่านทั้ง pHash ห่างไม่เกิน 4, dHash ห่างไม่เกิน 4 และค่าต่างเฉลี่ยของภาพย่อไม่เกิน 0.02 เจอ 4 คู่ ไม่มีคู่ไหน label ต่างกัน แล้วแบ่ง 5 fold ด้วย `StratifiedGroupKFold` (seed 42) โดยให้คู่ที่เจออยู่ fold เดียวกัน

### ขั้นที่ 8 เทียบ 5 สูตร

ทุกสูตรใช้ preprocessing และ geometry augmentation เหมือนขั้นที่ 5

| สูตร | โมเดล | epoch | lr backbone | lr head | schedule | สถานะ |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `friend_resnet18_fixed7` (baseline) | ResNet18 | 7 | 1e-4 | 1e-4 | คงที่ | ครบ 5 fold |
| `resnet18_cosine18` | ResNet18 | 18 | 1e-4 | 1e-4 | cosine, warmup 2 epoch | ครบ 5 fold |
| `resnet18_head_lr18` | ResNet18 | 18 | 5e-5 | 5e-4 | cosine, warmup 2 epoch | ครบ 5 fold |
| `efficientnet_b0_cosine18` | EfficientNet-B0 | 18 | 1e-4 | 3e-4 | cosine, warmup 2 epoch | หยุดกลาง fold 5 |
| `convnext_tiny_cosine18` | ConvNeXt-Tiny | 18 | 5e-5 | 3e-4 | cosine, warmup 2 epoch | ยังไม่ได้รัน |

cosine schedule คือค่อยๆ เพิ่ม learning rate ใน 2 epoch แรก แล้วค่อยๆ ลดลงเป็นรูปโค้งจนเกือบศูนย์ตอน epoch สุดท้าย

ผลเฉลี่ย 5 fold:

| สูตร | ไม่มี TTA | มี TTA | ต่างจาก baseline (มี TTA) |
| --- | ---: | ---: | ---: |
| `friend_resnet18_fixed7` | 0.94619 | 0.94897 | +0.0028 |
| **`resnet18_cosine18`** | 0.97369 | **0.97551** | **+0.0293** |
| `resnet18_head_lr18` | 0.97025 | 0.97243 | +0.0262 |

ผลราย fold (ไม่มี TTA / มี TTA):

| fold | fixed7 | cosine18 | head_lr18 |
| ---: | --- | --- | --- |
| 1 | 0.94216 / 0.94457 | 0.97287 / 0.97370 | 0.97313 / 0.97384 |
| 2 | 0.94183 / 0.94578 | 0.97469 / 0.97788 | 0.97175 / 0.97341 |
| 3 | 0.95086 / 0.95615 | 0.97162 / 0.97528 | 0.96882 / 0.97117 |
| 4 | 0.94528 / 0.94602 | 0.97547 / 0.97502 | 0.96841 / 0.97296 |
| 5 | 0.95083 / 0.95230 | 0.97381 / 0.97567 | 0.96914 / 0.97076 |

- สูตรที่ผ่านกติกาและคะแนนสูงสุดคือ `resnet18_cosine18` + TTA ดีกว่า baseline ทุก fold
- โมเดลเหมือนเดิมทุกอย่าง แค่เทรนนานขึ้นด้วย cosine schedule ก็ได้เพิ่มเกือบ 0.03
- baseline ได้ 0.946 ไม่ใช่ 0.954 แบบขั้นที่ 5 เพราะ fold ต่างกันและไม่ได้เลือก epoch จาก validation
- EfficientNet-B0 ได้ผลบาง fold ก่อนหยุด เช่น fold 3 ได้ 0.9764 / 0.9793 และ fold 4 ได้ 0.9799 / 0.9796 สูงกว่า cosine18 ใน fold เดียวกัน แต่ยังไม่ครบ 5 fold จึงไม่ได้เข้ารอบเลือก

### ขั้นที่ 9 Test-time augmentation (TTA)

ตอน predict ให้โมเดลดูภาพเดียวกัน 5 แบบ คือภาพเดิม, หมุน −8° และ +8°, ย่อ 0.95 เท่า และขยาย 1.05 เท่า แล้วเฉลี่ยความน่าจะเป็นก่อนเลือกคลาส ไม่ต้องเทรนใหม่

| fold | ไม่มี TTA | มี TTA | ต่าง |
| ---: | ---: | ---: | ---: |
| 1 | 0.97287 | 0.97370 | +0.0008 |
| 2 | 0.97469 | 0.97788 | +0.0032 |
| 3 | 0.97162 | 0.97528 | +0.0037 |
| 4 | 0.97547 | 0.97502 | −0.0005 |
| 5 | 0.97381 | 0.97567 | +0.0019 |

TTA เพิ่มเฉลี่ย +0.0018 ดีขึ้น 4 จาก 5 fold และทุกสูตรได้ประโยชน์เหมือนกัน บน test, TTA เปลี่ยนคำทำนาย 1,249 จาก 25,889 รูป (4.8%)

### ขั้นที่ 10 วิเคราะห์ error

ใช้คำทำนาย out-of-fold ของ `resnet18_cosine18` + TTA ครบทั้ง 31,546 รูป

| คลาส | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| 7-11k | 0.98 | 0.99 | 0.99 |
| fiveN | 0.98 | 0.97 | 0.97 |
| freefried | 0.98 | 0.98 | 0.98 |
| pube | 0.98 | 0.97 | 0.98 |
| roblock | 0.96 | 0.96 | 0.96 |

accuracy รวม 0.98 roblock ยังอ่อนที่สุด แต่ขึ้นจาก 0.92 ในขั้นที่ 5 เป็น 0.96

error rate แยกตามลักษณะภาพ (แบ่งเป็น 4 ช่วงเท่าๆ กัน Q1 = ค่าต่ำสุด):

| ลักษณะภาพ | Q1 | Q2 | Q3 | Q4 |
| --- | ---: | ---: | ---: | ---: |
| ความสว่าง | 2.9% | 1.3% | 1.6% | 2.6% |
| contrast | 3.2% | 1.7% | 1.8% | 1.6% |
| ปริมาณขอบ/รายละเอียด (edge energy) | 4.0% | 1.9% | 1.4% | 1.1% |

- ภาพที่มีรายละเอียดน้อยผิดบ่อยที่สุด เกือบ 4 เท่าของภาพที่รายละเอียดเยอะ
- contrast ต่ำผิดบ่อยเกือบ 2 เท่า
- ความสว่างมีปัญหาทั้งสองปลาย ทั้งภาพมืดจัดและสว่างจัด
- โมเดลก็มั่นใจน้อยลงในกลุ่มเหล่านี้ (confidence เฉลี่ย 0.97 เทียบกับ 0.99)

### ขั้นที่ 11 เทรนบน train ทั้งหมดและสร้าง submission

- เทรน `resnet18_cosine18` บน train ทั้ง 31,546 รูป 18 epoch ด้วยสูตรเดิมทุกอย่าง loss ลดจาก 0.529 ใน epoch แรกเหลือ 0.0119 ใน epoch สุดท้าย ขั้นนี้ไม่มี validation คะแนนอ้างอิงคือ CV จากขั้นที่ 8
- predict test ด้วย TTA ได้จำนวนคลาส 7-11k 5,017, fiveN 5,579, freefried 3,764, pube 7,247, roblock 4,282 คอลัมน์ task4 นี้อยู่ในไฟล์ `kaggle_t12345_081771.csv` ซึ่งได้คะแนนรวม **0.81771** สูงสุดของกลุ่มตอนนี้
- export checkpoint เป็น `resnet18_cosine18_pretrained.pt` (42.7 MB) เก็บ weights, ค่า preprocessing และสูตรการเทรนไว้ด้วยกัน
- สัดส่วนคลาสที่ทายบน test ต่างจาก train: 7-11k เป็น 39% ของ train แต่ทายได้ 19% ของ test ส่วน pube 19% ของ train แต่ทายได้ 28% ยังไม่รู้ว่า test มีสัดส่วนต่างจริงหรือโมเดลเอนเอียง

### ขั้นที่ 12 ตรวจสอบผลซ้ำบนเครื่องอื่น

ทำบนเครื่อง RTX 3050 แยกจากเครื่องที่เทรน

- **checkpoint:** โหลด `resnet18_cosine18_pretrained.pt` แล้ว predict test ด้วย TTA เดียวกัน ตรงกับคอลัมน์ task4 ในไฟล์ 0.81771 25,883 จาก 25,889 แถว (99.98%) อีก 6 แถวน่าจะต่างจากการคำนวณทศนิยมต่างเครื่อง
- **ขั้นที่ 5:** รันซ้ำ fold 1 และ 2 ได้ 0.95246 และ 0.95528 ตรงทุกหลัก
- **ภาพเกือบซ้ำ:** เกณฑ์ในขั้นที่ 7 เข้มมาก จึงลองเกณฑ์ที่หลวมกว่า (pHash ไม่เกิน 4, dHash ไม่เกิน 16 เลือกจากการเปิดดูคู่ภาพ) รวมกลุ่มได้ 1,806 รูป เกือบทั้งหมดเป็น 7-11k พอวัดบน fold 1 ของขั้นที่ 5 พบว่ารูป validation ที่มีคู่อยู่ฝั่ง train ทายถูกมากกว่าจริง แต่ทำให้คะแนนสูงเกินจริงแค่ 0.00076 ไม่เปลี่ยนข้อสรุป (`src/task4/eda/build_eda_near_duplicate_v3.ipynb`)
- **TTA:** ลอง TTA แยกบนสูตรของขั้นที่ 5 ด้วยมุมที่แรงกว่าเล็กน้อย (±10°, 0.9/1.1) ได้ +0.0031 และ +0.0018 บน fold 1 และ 2 ไปทางเดียวกับขั้นที่ 9 (`src/task4/models/build_model_v3.ipynb`)
- เจอ 4 คู่ภาพ roblock กับ 7-11k ที่ผ่านเกณฑ์แต่ label ต่างกัน (`PLY11408`–`PLY02786`, `PLY11408`–`PLY19088`, `PLY03613`–`PLY04719`, `PLY06207`–`PLY30885`) ยังไม่ได้เปิดดู

## 3. สรุปผล

| ขั้น | วิธี | macro F1 |
| --- | --- | ---: |
| 3 | LightGBM บน feature ดั้งเดิม | 0.772 (holdout) |
| 5 | ResNet18 fine-tune + geometry | 0.954 (เฉลี่ย 3 fold) |
| 8–9 | ResNet18 cosine 18 epoch + TTA | **0.976** (เฉลี่ย 5 fold) |

ตัวเลขแต่ละแถวมาจาก validation คนละชุด จึงใช้ดูภาพรวมของการพัฒนาเท่านั้น การเทียบที่ยุติธรรมอยู่ในขั้นที่ 8 ซึ่งทุกสูตรใช้ fold เดียวกัน

สิ่งที่ช่วยมากที่สุดเรียงตามลำดับ: fine-tune pretrained CNN ทั้งโมเดล (+0.18), เทรนนานขึ้นด้วย cosine schedule (+0.03), geometry augmentation (+0.006) และ TTA (+0.002)

## 4. ข้อจำกัด

- สูตรสุดท้ายถูกเลือกจากคำทำนาย out-of-fold ชุดเดียวกับที่ใช้รายงาน 0.9755 ค่านี้จึงอาจสูงกว่าจริงเล็กน้อย แต่ส่วนต่าง +0.029 ใหญ่กว่าความแกว่งระหว่าง fold มาก
- การแบ่ง fold จับภาพเกือบซ้ำได้น้อย CV อาจสูงเกินจริงประมาณ 0.001
- EfficientNet-B0 ยังไม่ครบ 5 fold และ ConvNeXt-Tiny ยังไม่ได้รัน
- Kaggle แสดงแค่คะแนนรวม จึงดูผลของ Task 4 บน leaderboard แยกไม่ได้

## 5. งานถัดไป

1. รัน EfficientNet-B0 ให้ครบ 5 fold แล้วลอง ensemble กับ `resnet18_cosine18` ตามที่ follow-up notebook วางไว้
2. แก้ภาพ contrast ต่ำ รายละเอียดน้อย และมืดหรือสว่างจัด เช่นสุ่มปรับความสว่าง/contrast ตอนเทรน แล้ววัดบน fold เดียวกันก่อน
3. เปิดดู 4 คู่ roblock–7-11k ที่ label ไม่ตรงกัน

## 6. ไฟล์ที่เกี่ยวข้อง

| ขั้น | ไฟล์ |
| --- | --- |
| 1 | `src/task4/eda/build_eda_original.ipynb` |
| 2 | `src/task4/eda/build_eda_1.ipynb`, `src/task4/image_features.py` |
| 3 | `src/task4/models/build_model.ipynb` |
| 4 | `src/task4/eda/build_eda_original_v2.ipynb`, `src/task4/eda/build_eda_1_v2.ipynb` |
| 5–6 | `src/task4/models/build_model_v2.ipynb`, `src/task4/models/image_training.py` |
| 7–11 | `Kaggle-Final/of_friends/Task4_gpu_followup.ipynb`, `Kaggle-Final/of_friends/resnet18_cosine18_pretrained.pt` (นอก repo) |
| 11 | `Kaggle-Final/kaggle_t12345_081771.csv` (นอก repo) |
| 12 | `src/task4/eda/build_eda_near_duplicate_v3.ipynb`, `src/task4/models/build_model_v3.ipynb`, `src/task4/models/tta.py` |
