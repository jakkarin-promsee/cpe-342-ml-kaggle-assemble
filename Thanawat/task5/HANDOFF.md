# Task5 Handoff — Account Security Monitoring

## รุ่นหลัก: density_auto_v1

ชุดส่งสุดท้ายใช้ **Task1–4 ชุดที่เลือก** และ **Task5 รุ่น `density_auto_v1`** ได้คะแนนรวม Kaggle Public **0.82864** รุ่น Task5 เลือก 2,046 บัญชีเป็น alert โดยใช้ `1` สำหรับ anomaly และ `0` สำหรับบัญชีอื่น

ข้อมูลมี 25,889 บัญชี ตัวแปรเชิงตัวเลข 122 ตัว และ `player_id` สำหรับตรวจการจับคู่แถว ตัวแปรพฤติกรรมหลายกลุ่มมีข้อมูลสี่ช่วงเวลา เช่น การล็อกอิน อุปกรณ์ พิกัด และการซื้อ

## สามองค์ประกอบของคะแนน

1. **Within-account Change** จับบัญชีที่พฤติกรรมเปลี่ยนจากประวัติของตัวเอง สรุประดับและการเปลี่ยนแปลงของแต่ละกลุ่มตัวแปรเป็น 30 level features และ 28 shift features โดยเปรียบเทียบช่วงที่ 4 กับช่วงที่ 1–3 และครึ่งหลังกับครึ่งแรก ใช้ median/MAD ลดผลจากค่าที่สุดโต่ง แล้วเทียบความแรงของการเปลี่ยนกับข้อมูลอ้างอิงที่สลับค่าในแต่ละคอลัมน์ พิกัดใช้ระยะ Haversine สูงสุดระหว่างคู่พิกัดที่ถูกต้อง
2. **Univariate Density Modes** จับกลุ่มย่อยที่มีร่องความหนาแน่นคั่นจากกลุ่มหลักในตัวแปรเดียว ใช้ histogram ที่ปรับให้เรียบเพื่อหากลุ่มเหล่านี้ ซึ่งอาจอยู่กลางช่วงข้อมูล ข้ามตัวแปรแบบ discrete ที่มีค่าต่างกันน้อยและไม่เหมาะกับการหาร่องความหนาแน่น
3. **Cross-feature Dependence** จับความผิดปกติที่อาจเห็นเมื่อดูหลายตัวแปรร่วมกัน เลือกบัญชีที่มีคะแนน shift ต่ำกว่า 0.65 และ island score เท่ากับศูนย์เป็นกลุ่มอ้างอิง ได้ 24,098 บัญชี ฝึก HistGradientBoostingClassifier ให้แยกแถวจริงออกจากแถวที่สลับค่าแต่ละคอลัมน์ การสลับรักษาการกระจายของแต่ละตัวแปรไว้แต่ทำลายความสัมพันธ์ระหว่างตัวแปร ใช้ 5 folds × 3 bags รวม 15 classifiers และสร้าง 140 sparse projections จากตัวแปรครั้งละ 2–3 ตัว

แต่ละส่วนตรวจความผิดปกติคนละแบบ ทั้งการเปลี่ยนจากประวัติ กลุ่มย่อยในตัวแปรเดียว และรูปแบบร่วมระหว่างตัวแปร หลังปรับสเกลคะแนนแล้ว รุ่นนี้รวมสัญญาณด้วย noisy-OR:

`score = 1 - (1 - p_shift) × (1 - p_island) × (1 - p_dependence)`

คะแนนที่ได้ใช้จัดอันดับ และเลือกจำนวน alerts ด้วยค่าประมาณ Macro F3 จากผลรวมคะแนนของโมเดล จุดตัดนี้ให้ 2,046 alerts ในข้อมูลชุดนี้ สมมติฐานของ noisy-OR อาจคลาดเคลื่อนเมื่อสัญญาณสัมพันธ์กัน และคะแนนแต่ละส่วนยังไม่ได้ปรับเทียบกับป้ายกำกับจริง

## ผลตรวจและข้อจำกัด

- การทดลองแยกองค์ประกอบให้ 810 alerts เมื่อใช้ Within-account Change อย่างเดียว, 981 เมื่อใช้ Univariate Density Modes อย่างเดียว และ 1,791 เมื่อรวมสองส่วน เทียบกับ 2,046 ของรุ่นหลัก จำนวนเหล่านี้บอกว่าการตัดสินเปลี่ยนไปเพียงใด
- AUC เฉลี่ยของ HGB ในการแยกแถวจริงจากแถวที่สลับค่าอยู่ที่ 0.5086 ใกล้การสุ่ม จึงยังไม่มีหลักฐานจากการทดสอบนี้ว่าการเพิ่มองค์ประกอบที่สามช่วยตรวจ anomaly จริง
- รุ่นควบคุมใช้ ranking เดียวกันแต่เลือก 6,531 alerts ส่วน conditional residual ที่ทำนายพฤติกรรมช่วงที่ 4 จากช่วงที่ 1–3 ใช้เพื่อวิเคราะห์ประกอบและไม่ได้รวมในคะแนนรุ่นหลัก
- ตัว classifier ให้คะแนนแบบ cross-fitting และ fit imputer/scaler กับกลุ่มอ้างอิงของ fold ฝึก ส่วนการสร้าง geometry การเลือกกลุ่มอ้างอิง และ calibration บางขั้นใช้ข้อมูลทั้งชุดโดยไม่มี labels วิธีนี้เป็น transductive และยังต้องตรวจว่าข้ออนุญาตสำหรับ Task5 ครอบคลุมการใช้ข้อมูลชุดนี้ในขั้นตอนดังกล่าว

## วิธีอ่านและรัน notebook

เริ่มจาก `Task5_density_ratio_followup.ipynb` ซึ่งอธิบายสามองค์ประกอบของรุ่นหลัก ตรวจผลแยกองค์ประกอบ และเปรียบเทียบจุดตัดแบบอัตโนมัติกับจุดตัด 6,531 alerts อ่านผลโดยดูจำนวนบัญชีที่ถูกเลือกและความไวต่อการเปลี่ยนวิธีรวมคะแนน ควบคู่กับข้อจำกัดของข้อมูลที่ไม่มีป้ายกำกับ

ชุดเอกสารส่งประกอบด้วย notebooks และ handoff นี้ ตัว notebook ของรุ่นหลักตั้งค่าเริ่มต้นให้อ่านผลที่บันทึกไว้ การกด Run All จากชุดเอกสารส่งเพียงอย่างเดียวจึงไม่เท่ากับการรันซ้ำครบตั้งแต่รับข้อมูลจนได้คำทำนาย เนื้อหาใน handoff นี้สรุปวิธีและผลที่รันแล้วเพื่อให้อ่านได้ในตัว

หากตรวจชุดคำทำนายของรุ่นหลัก ให้นับ 25,889 แถว ตรวจ ID ไม่ซ้ำ และตรวจว่าคอลัมน์ Task5 มีเพียง 0 และ 1 โดยมี 1 จำนวน 2,046 แถว

## การทดลองต่อยอดที่แนบมาด้วย

- `Task5_residual_cluster_followup.ipynb` ทดลอง conditional quantile residual และ ECDF/UMAP/HDBSCAN เพื่อดูว่าการคาดพฤติกรรมจากบริบทและการจับกลุ่มให้สัญญาณต่างจาก density ranking แค่ไหน ผลชุดนี้ใช้เปรียบเทียบเชิงสำรวจ
- `Task5_rank_fusion_followup.ipynb` ผสม percentile rank ของ Density auto 95% กับ conditional residual 5% ที่ 2,046 alerts ทดลองแล้วคะแนนรวมต่ำกว่ารุ่นหลัก จึงคง `density_auto_v1`
- `Task5_density_context_followup.ipynb` ทดสอบสามกรณีแยกกัน ได้แก่ เพิ่ม alerts เป็น 2,250 บน ranking เดิม เพิ่มเป็น 5 bags ที่ 2,046 alerts และเปลี่ยน HGB เป็น conditional permutation ที่ 2,046 alerts ใช้ผลเปรียบเทียบเพื่ออ่านความไวของวิธี ยังไม่มีคะแนน Public ที่บันทึกไว้สำหรับสามกรณีนี้

Notebook การทดลองบางเล่มเปิดผลที่บันทึกไว้ การรันใหม่ต้องมีข้อมูล โค้ดและผลฝึกที่เกี่ยวข้องตาม setup ในแต่ละเล่ม.

## รายการ notebooks ทั้งหมด

รวมรุ่นหลัก สมุด EDA และการทดลองย้อนหลังตามตำแหน่งในโฟลเดอร์ Task5 บางเล่มต้องใช้ข้อมูลและผลฝึกเดิมเพื่อรันซ้ำ

- `Task5_behavior_blocks_followup.ipynb`
- `Task5_cut_study.ipynb`
- `Task5_density_context_followup.ipynb`
- `Task5_density_ratio_followup.ipynb`
- `Task5_directional_budget_followup.ipynb`
- `Task5_masked_temporal_followup.ipynb`
- `Task5_rank_fusion_followup.ipynb`
- `Task5_representation_study.ipynb`
- `Task5_residual_cluster_followup.ipynb`
- `Task5_subspace_cluster_audit.ipynb`
- `Task5_tail_family_followup.ipynb`
- `Task5_trajectory_mixture_followup.ipynb`
- `eda/build_eda_1.ipynb`
- `eda/build_eda_1_v2.ipynb`
- `eda/build_eda_behavior_v3.ipynb`
- `eda/build_eda_original.ipynb`
- `eda/build_eda_original_v2.ipynb`
- `eda/inspect_decision_lines_v1.ipynb`
- `models/build_model.ipynb`
- `models/build_model_v2.ipynb`
- `models/build_model_v3.ipynb`
- `models/build_model_v4.ipynb`
- `models/build_model_v5.ipynb`
- `models/build_model_v6.ipynb`
- `models/build_model_v7.ipynb`
- `models/rank_diagnostics_v1.ipynb`
