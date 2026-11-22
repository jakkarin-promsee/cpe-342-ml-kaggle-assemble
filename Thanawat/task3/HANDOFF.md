# Task3 Handoff — Monthly Spending Prediction

## รุ่นหลัก

ใช้ **Conditional Quantile Blend** จาก `Task3_native_quantile_followup.ipynb` ผสมผลทำนายเดิมกับ CatBoost native categorical แบบ **50/50** ผลทำนายเป็นยอดใช้จ่าย 30 วันหน่วยบาท การแข่งขันใช้ NMAE ส่วนตัวเลข validation ด้านล่างเป็น **MAE หน่วยบาท**

## ข้อมูลและวิธีทำนาย

Train มี 104,000 แถว, Test 25,889 แถว ใช้ 32 raw features และ 16 features ที่สร้างเพิ่ม ยอดใช้จ่ายมีค่าศูนย์จำนวนหนึ่ง จึงแยกโมเดลเป็นสองขั้น

1. ใช้ CatBoost classifier ประมาณโอกาสใช้จ่ายมากกว่าศูนย์ `p` ถ้า `p ≤ 0.5` ทำนาย 0
2. ถ้า `p > 0.5` ประมาณ median ของยอดใช้จ่ายรวมศูนย์ด้วย conditional quantile ที่ระดับ `1 − 0.5/p` ของยอดบวก ประมาณ quantiles บน log1p ของยอดบวก แปลงกลับเป็นบาทและ interpolate ระหว่าง quantiles

`Task3_hurdle_followup.ipynb` อธิบายและสร้างตัวอ้างอิงเดิม รุ่นใหม่คง classifier เดิมและเปลี่ยนตัวทำนาย quantile ให้จัด categorical โดยตรง ใช้ CatBoost 900 trees, depth 5, learning rate 0.04 และ L2 = 10 จากนั้นเฉลี่ยผลทำนายสุดท้ายของรุ่นเก่าและใหม่

## การประเมิน

| ชุดประเมิน | รุ่นเดิม | Blend 50/50 |
| --- | ---: | ---: |
| MAE เฉลี่ย 5-fold OOF บน development 83,200 แถว | 2,789.82 | **2,770.30** |
| MAE บน holdout ที่ใช้ซ้ำ 20,800 แถว | 2,769.97 | **2,756.49** |

Blend ชนะครบ 5 folds ส่วน LightGBM แบบทำนายบาทโดยตรง แบบ raw positive quantiles และ OOF calibration ไม่ผ่านเกณฑ์เลือก หลังผล holdout ดีขึ้นจึง refit บน Train ทั้งหมด และได้คะแนนรวม Public **0.81055** จาก **0.81047** ในการเปรียบเทียบที่เปลี่ยนเฉพาะ Task3 ผลจาก holdout ที่ใช้ซ้ำเป็นหลักฐานเปรียบเทียบภายในโครงการ

## วิธีอ่านและตรวจคำตอบ

เริ่มจาก notebook รุ่นหลักเพื่ออ่านผลครบทุก candidate และใช้ notebook hurdle เมื่อต้องการดูที่มาของสูตร conditional median โหมดปกติแสดงผลที่คำนวณไว้ การรันใหม่ต้องมีข้อมูล โมดูลของโปรเจกต์และผล baseline ตรวจว่า `task3` เป็นตัวเลข finite ไม่ติดลบ และมี 25,889 ค่าตามลำดับ Test
