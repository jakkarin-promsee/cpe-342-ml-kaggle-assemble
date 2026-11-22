# Task4 Handoff — Game Title Detection

## รุ่นหลัก

ใช้ **EfficientNet Blend 50** จาก `Task4_backbone_diversity.ipynb` รวม probability ของ EfficientNet-B0 50%, ResNet18 สูตร compound corruption 25% และ ResNet18 สูตร sampler 25% แล้วเลือกคลาสที่มี probability สูงสุด Metric คือ **Macro F1**

ส่งรหัสคลาสเป็นตัวเลข 0 = 7-Eleven Knight, 1 = FiveN, 2 = Free Fried, 3 = PUBE Mobile และ 4 = Roblock

## การฝึกและจุดที่ต่างจากรุ่นเดิม

- สูตร sampler มาจากการทดลอง class imbalance และ strong augmentation ใน `Task4_heavy_balance.ipynb` ใช้การ sampling เพื่อเพิ่มโอกาสเห็นคลาสที่มีตัวอย่างน้อยขณะฝึก
- `Task4_compound_supervised.ipynb` ทดสอบการรบกวนภาพหลายอย่างร่วมกัน เช่น หมุนร่วมกับความมืดหรือ noise เพื่อฝึกความทนทานต่อภาพที่เสื่อมหลายแบบพร้อมกัน
- รุ่นหลักเพิ่ม EfficientNet-B0 ที่ใช้ pretrained initialization และสูตร compound 18 epochs เพื่อให้ ensemble มี backbone อีกแบบ ทดสอบทั้ง EfficientNet เดี่ยวและสูตรผสมบน validation rows ชุดเดียวกัน

## ผลประเมินและการเลือก

ใช้ 5 folds จาก Train เพื่อเทียบภาพเดิมและ stress suite จากภาพ validation ที่สังเคราะห์การรบกวนขึ้น การเทียบกับคู่ ResNet เดิมให้ Macro F1 ของ blend เพิ่ม **0.00307** บนภาพเดิม และ **0.00790** บน stress แบบถ่วงน้ำหนักเท่ากันราย family ชนะ stress **5/5 folds** ส่วน audit เงื่อนไขเพิ่มเติมได้ mean gain **0.01811** การประเมินใช้ภาพ validation ที่เคยใช้ในรอบก่อนและการรบกวนภาพแบบจำลอง จึงยังมีข้อจำกัดในการประเมินความทนทานต่อ Test จริง

เลือก blend ก่อนทำนาย Test, ใช้ full-train weights ของแต่ละสมาชิก และใช้ test-time augmentation ตามสูตรที่กำหนดไว้ คะแนนรวมของชุดส่ง EfficientNet blend ที่รายงานคือ **0.81047** ตัวเลขนี้รวมทั้งห้า Tasks

## วิธีอ่านและตรวจคำตอบ

อ่าน notebook รุ่นหลักก่อนเพื่อดู setup, OOF, เกณฑ์เลือก, audit และ full refit ส่วนสอง notebook ของ ResNet อธิบายที่มาของสมาชิกเดิม การรันครบต้องมีชุดภาพและ weights หรือผลฝึกของสมาชิกเดิมตามขั้นตอน การฝึกใหม่หลาย folds ใช้ GPU และเวลามาก ตรวจให้ `task4` เป็นรหัส 0–4 ครบ 25,889 แถว และใช้ class mapping เดียวกันก่อนเฉลี่ย probability ของทุกสมาชิก

## รายการ notebook ทั้งหมด

รวม notebook ของ Task4 ตามโครงสร้างที่คัดลอกมา รายการนี้ครอบคลุม EDA, baseline, pilot, การทดลองเทียบและรุ่นที่ใช้ส่ง บางเล่มต้องใช้ข้อมูล โค้ดประกอบ หรือผลฝึกจากโปรเจกต์เดิม

- `Task4_advanced_eda.ipynb`
- `Task4_augmix_consistency.ipynb`
- `Task4_backbone_diversity.ipynb`
- `Task4_class_bias.ipynb`
- `Task4_class_specific_stacking.ipynb`
- `Task4_compound_supervised.ipynb`
- `Task4_convnext_diversity.ipynb`
- `Task4_curriculum_restart.ipynb`
- `Task4_denoise_pilot.ipynb`
- `Task4_denoise_submission.ipynb`
- `Task4_ensemble_weights.ipynb`
- `Task4_fourier_expert.ipynb`
- `Task4_fullframe_pilot.ipynb`
- `Task4_gpu_followup.ipynb`
- `Task4_heavy_balance.ipynb`
- `Task4_mixed_consistency.ipynb`
- `Task4_mixup_pilot.ipynb`
- `Task4_noise_supervised.ipynb`
- `Task4_quality_router.ipynb`
- `Task4_seed_diversity.ipynb`
- `Task4_teacher_student.ipynb`
- `eda/build_eda_1.ipynb`
- `eda/build_eda_1_v2.ipynb`
- `eda/build_eda_original.ipynb`
- `eda/build_eda_original_v2.ipynb`
- `models/build_model.ipynb`
- `models/build_model_v2.ipynb`
