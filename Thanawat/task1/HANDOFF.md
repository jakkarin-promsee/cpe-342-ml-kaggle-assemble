# Task1 Handoff — Anti-Cheat Pre-Filter

## รุ่นหลัก

ใช้ **LightGBM slow800_bagging** จาก `Task1.ipynb` เพื่อจำแนกบัญชีที่ถูก flag ไว้แล้ว ให้คำตอบ 0 = Legitimate และ 1 = Cheater ประเมินด้วย **Macro F2** รุ่นที่เลือกได้ค่าเฉลี่ย **0.77911** จาก 3 folds × 2 repeats และใช้ threshold **0.375** หลัง refit

## ข้อมูลและการเตรียม

Train มี 104,000 แถวและ predictors เชิงตัวเลข 31 ตัว มี target 90,816 แถวและไม่มี target 13,184 แถว ใช้เฉพาะแถวที่มี target ในงาน supervised และเก็บ missing ของ predictors ให้ LightGBM จัดการ แยก player ID ออกจาก predictors ข้อมูล Test มี 25,889 แถว

EDA ใช้ Train เพื่อดู distribution แยกคลาส, missing rate, correlation และความสัมพันธ์ตามอายุบัญชี ตัวเลขสูงหรือต่ำรายตัว เช่น win rate หรือ reports received ไม่พอที่จะสรุปว่า label ผิด โดยเฉพาะเมื่อข้อมูลมาจากกลุ่มบัญชีที่ระบบเดิมคัดมาแล้ว

## การเลือกโมเดล

- รุ่นหลักใช้ 24 raw features ตัด utility_usage_rate, sessions_per_day, first_blood_rate, avg_session_length_min, clutch_success_rate, input_consistency_score และ reaction_time_ms คง account_age_days และ friend_network_size
- ใช้ 800 trees, learning rate 0.02, row sampling 0.8 ทุก iteration และ feature sampling 0.9
- เลือก threshold ด้วย inner 3-fold CV ภายในชุดฝึกของแต่ละ outer fold ก่อนวัดคะแนนบน outer validation รุ่น 24 features ที่ใช้ 300 trees ได้ **0.77819** และการเพิ่มผลคูณหรือ joint encoding จาก pairs/triples ไม่เพิ่มคะแนนเฉลี่ยในรอบที่ทดลอง
- การเปรียบเทียบใช้ folds ที่เคยใช้สำรวจและเลือก features มาแล้ว จึงเป็นผลเปรียบเทียบเชิงทดลอง

## วิธีรันและตรวจคำตอบ

เปิด `Task1.ipynb` ด้วย environment ที่มี NumPy, pandas, matplotlib, scikit-learn, LightGBM, joblib และ pyarrow วางข้อมูลการแข่งขัน (Train, Test และ datacard) ใน `data/raw` ใต้โฟลเดอร์ notebook หรือตั้ง `TASK1_RAW_DIR` แล้วกด **Run All**

ส่วน 4.3 มีตัวเลือกเปิดการสำรวจ combinations ทั้งหมดและ joint permutation importance แยกกัน ส่วน 5–9 ฝึก เปรียบเทียบ OOF และ refit รุ่นที่เลือก เมื่อไม่มี cache จะเริ่มฝึกใหม่จาก Train ใน notebook เดียว ส่วน 10 ส่งออก `id,task1` ไปยัง `outputs/task1_predictions.csv` และตรวจ 25,889 แถวตามลำดับ Test โดย `task1` เป็น 0 หรือ 1 ส่วน 11 แสดงรุ่นที่เลือก, ค่าเฉลี่ย Macro F2, threshold และจำนวนคำตอบแยกคลาส
