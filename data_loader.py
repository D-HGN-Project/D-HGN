"""
D-HGN数据加载器
支持: fMRI动态FC + DTI结构连接(SC)矩阵 多模态加载
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold


class DHGNDataLoader:
    def __init__(self, data_root="./data"):
        self.data_root = data_root
        self.num_rois = 90
        self.num_timepoints = 130
        self.window_size = 60
        self.stride = 1
        self.num_windows = self.num_timepoints - self.window_size + 1

    def load_all_data(self, groups=['EMCI', 'CN'], use_sc=True):
        """
        加载多模态数据
        Args:
            groups: 要加载的组别
            use_sc: 是否使用SC矩阵替代非成像数据
        Returns:
            dynamic_graphs, sc_matrices (或 non_imaging), labels, subject_ids
        """
        all_dynamic = []
        all_sc = []  # SC矩阵 (替代non_imaging)
        all_non_imaging = []  # 备用: 年龄+性别
        all_labels = []
        all_subject_ids = []

        for group in groups:
            csv_path = os.path.join(self.data_root, f"{group}.csv")
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(csv_path, encoding='gbk')
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_path, encoding='iso-8859-1')

            if group == 'EMCI':
                start_id, end_id = 100, 178  # EMCI: sub100-sub178
                group_label = 1
            elif group == 'CN':
                start_id, end_id = 71, 141  # CN: sub071-sub141
                group_label = 0
            elif group == 'AD':
                start_id, end_id = 1, 70  # AD: sub001-sub070
                group_label = 1  # AD vs CN时，AD也是label=1（正类）

            for idx in range(start_id, end_id + 1):
                subject_id = f"sub{idx:03d}"
                # 支持多种列名格式
                if 'subject_id' in df.columns:
                    subject_row = df[df['subject_id'] == subject_id]
                elif 'Subject_ID' in df.columns:
                    subject_row = df[df['Subject_ID'] == subject_id]
                else:
                    print(f"⚠️  找不到subject_id列")
                    continue
                if subject_row.empty:
                    continue

                try:
                    # 1. 加载 fMRI 动态FC
                    dfc_path = os.path.join(self.data_root, group, 'GretnaDFCMatrixZ', f'z{subject_id}.mat')
                    dynamic_fc = self._load_dynamic_fc(dfc_path)

                    # 验证并调整DFC形状
                    expected_windows = self.num_windows
                    expected_rois = self.num_rois
                    
                    # 如果窗口数超过预期，静默截取
                    if dynamic_fc.shape[0] > expected_windows:
                        dynamic_fc = dynamic_fc[:expected_windows, :, :]
                    elif dynamic_fc.shape[0] < expected_windows:
                        print(f"⚠️  剔除 {subject_id}: DFC窗口不足 {dynamic_fc.shape[0]} < {expected_windows}")
                        continue
                    
                    # 检查ROI维度
                    if dynamic_fc.shape[1] != expected_rois or dynamic_fc.shape[2] != expected_rois:
                        print(f"⚠️  剔除 {subject_id}: DFC ROI维度不一致 {dynamic_fc.shape}")
                        continue

                    if np.isnan(dynamic_fc).any() or np.isinf(dynamic_fc).any():
                        print(f"⚠️  剔除 {subject_id}: DFC包含NaN或Inf")
                        continue

                    # 2. 加载 DTI SC矩阵 (使用ADNI Subject ID匹配)
                    sc_matrix = None
                    if use_sc:
                        sc_dir = os.path.join(self.data_root, group, 'Final_SC_Matrices')
                        # 今CSV获取ADNI Subject ID (e.g., "002_S_4473" -> "S_4473")
                        adni_subject = None
                        if 'Subject' in subject_row.columns:
                            full_adni = str(subject_row['Subject'].values[0])
                            # 提取S_XXXX部分
                            if '_S_' in full_adni:
                                adni_subject = 'S_' + full_adni.split('_S_')[1]
                            else:
                                adni_subject = full_adni
                        sc_matrix = self._load_sc_matrix(sc_dir, subject_id, adni_subject)
                        if sc_matrix is None:
                            print(f"⚠️  剔除 {subject_id}: 未找到SC矩阵")
                            continue
                        if np.isnan(sc_matrix).any() or np.isinf(sc_matrix).any():
                            print(f"⚠️  剔除 {subject_id}: SC包含NaN或Inf")
                            continue

                    # 3. 备用: 年龄+性别
                    age = float(subject_row['Age'].values[0])
                    sex = 1.0 if subject_row['Sex'].values[0] == 'M' else 0.0

                    all_dynamic.append(dynamic_fc)
                    if use_sc and sc_matrix is not None:
                        all_sc.append(sc_matrix)
                    all_non_imaging.append([age, sex])
                    all_labels.append(group_label)
                    all_subject_ids.append(subject_id)

                except Exception as e:
                    print(f"❌ 加载 {subject_id} 失败: {e}")
                    continue

        if not all_dynamic:
            print("警告: 未加载到任何数据！")
            return None, None, None, None

        dynamic_graphs = np.array(all_dynamic)
        labels = np.array(all_labels)

        # 🔧 DFC预处理：处理Gretna的特殊标记值
        print(f"\n🔧 数据预处理:")
        print(f"  DFC原始范围: [{dynamic_graphs.min():.4f}, {dynamic_graphs.max():.4f}]")
        n_special = np.sum(dynamic_graphs > 10)
        total_values = dynamic_graphs.size
        print(f"  → 检测到 {n_special:,} 个特殊标记 (占比 {n_special/total_values*100:.2f}%)")
        dynamic_graphs[dynamic_graphs > 10] = 0
        print(f"  → 已替换为0, 当前范围: [{dynamic_graphs.min():.4f}, {dynamic_graphs.max():.4f}]")

        non_imaging = np.array(all_non_imaging, dtype=np.float32)
        if len(non_imaging) > 0:
            age_mean = np.mean(non_imaging[:, 0])
            age_std = np.std(non_imaging[:, 0]) + 1e-6
            non_imaging[:, 0] = (non_imaging[:, 0] - age_mean) / age_std
        self.non_imaging_data = non_imaging

        # 根据模式选择返回SC矩阵或非成像数据
        if use_sc and all_sc:
            sc_matrices = np.array(all_sc)
            # SC矩阵归一化 (每个被试单独归一化到0-1)
            for i in range(len(sc_matrices)):
                max_val = sc_matrices[i].max()
                if max_val > 0:
                    sc_matrices[i] = sc_matrices[i] / max_val
            print(f"  SC矩阵形状: {sc_matrices.shape}")
            modal_data = sc_matrices
        else:
            print(f"  使用非成像数据 (年龄+性别)")
            modal_data = non_imaging

        print(f"\n✅ 最终数据:")
        print(f"  样本数: {len(all_subject_ids)}")
        disease_group = [g for g in groups if g != 'CN'][0] if len(groups) > 1 else 'Unknown'
        print(f"  标签分布: CN={np.sum(labels==0)}, {disease_group}={np.sum(labels==1)}")

        return dynamic_graphs, modal_data, labels, all_subject_ids

    def _load_dynamic_fc(self, mat_path):
        try:
            mat_data = sio.loadmat(mat_path)
            if 'DZStruct' in mat_data:
                dz_struct = mat_data['DZStruct']
                fields = dz_struct.dtype.names
                w_fields = [f for f in fields if f.startswith('W_')]
                w_fields.sort(key=lambda x: int(x.split('_')[1]))

                dynamic_matrices = []
                for field in w_fields:
                    matrix = dz_struct[0, 0][field]
                    if matrix.shape == (self.num_rois, self.num_rois):
                        dynamic_matrices.append(matrix)

                if dynamic_matrices:
                    return np.stack(dynamic_matrices, axis=0)

            for key in ['DFC', 'dfc', 'dynamic_fc', 'result', 'data']:
                if key in mat_data:
                    data = mat_data[key]
                    if isinstance(data, np.ndarray) and data.ndim == 3:
                        return data

            for key, value in mat_data.items():
                if key.startswith('__'): continue
                if isinstance(value, np.ndarray):
                    if value.ndim == 3 and value.shape[1] == value.shape[2]:
                        return value
                    if value.ndim == 4 and value.shape[0] == 1:
                        return value[0]

            return np.zeros((self.num_windows, self.num_rois, self.num_rois))
        except Exception:
            return np.zeros((self.num_windows, self.num_rois, self.num_rois))

    def _load_sc_matrix(self, sc_dir, subject_id, adni_subject=None):
        """
        加载DTI结构连接矩阵
        使用ADNI Subject ID精确匹配SC文件
        
        Args:
            sc_dir: SC矩阵文件夹
            subject_id: 内部ID (如 sub100)
            adni_subject: ADNI ID (如 S_4473)，用于文件名匹配
        """
        try:
            if not os.path.exists(sc_dir):
                return None
            
            # 获取所有SC文件
            sc_files = [f for f in os.listdir(sc_dir) if f.endswith('number_of_tracts.connectivity.mat')]
            if not sc_files:
                return None
            
            matched_file = None
            
            # 方法1：使用ADNI Subject ID精确匹配
            if adni_subject:
                for sc_file in sc_files:
                    # SC文件名中包含 S_XXXX 模式
                    if adni_subject in sc_file:
                        matched_file = sc_file
                        break
            
            # 方法2：如果精确匹配失败，回退到顺序索引（兼容旧逻辑）
            if matched_file is None:
                sorted_files = sorted(sc_files)
                idx = int(subject_id.replace('sub', ''))
                
                if idx >= 219:  # AD
                    relative_idx = idx - 219
                elif idx >= 105:  # CN
                    relative_idx = idx - 105
                else:  # EMCI
                    relative_idx = idx - 1
                
                if 0 <= relative_idx < len(sorted_files):
                    matched_file = sorted_files[relative_idx]
            
            if matched_file is None:
                return None
            
            # 加载.mat文件
            mat_path = os.path.join(sc_dir, matched_file)
            mat_data = sio.loadmat(mat_path)
            
            # 提取connectivity矩阵
            if 'connectivity' in mat_data:
                sc_matrix = mat_data['connectivity']
                if sc_matrix.ndim == 2:
                    # 如果SC矩阵是116×116，截取前90×90以匹配fMRI ROI
                    if sc_matrix.shape[0] > self.num_rois:
                        sc_matrix = sc_matrix[:self.num_rois, :self.num_rois]
                    return sc_matrix.astype(np.float32)
            
            return None
        except Exception as e:
            print(f"加载SC矩阵失败 ({subject_id}): {e}")
            return None

    def _load_static_fc(self, txt_path):
        """加载并预处理静态FC"""
        try:
            static_fc = np.loadtxt(txt_path)
            
            # 应用与动态FC相同的预处理
            # 步骤1：处理Gretna特殊标记
            static_fc[static_fc > 10] = 0
            
            # 步骤2：Clip异常值
            static_fc = np.clip(static_fc, -2, 2)
            
            # 步骤3：Z-score归一化
            mean_val = np.mean(static_fc)
            std_val = np.std(static_fc) + 1e-8
            static_fc = (static_fc - mean_val) / std_val
            
            return static_fc
        except Exception as e:
            print(f"警告: 加载静态FC失败 ({txt_path}): {e}")
            return np.zeros((self.num_rois, self.num_rois))


    def data_split(self, labels, n_folds=10):
        if n_folds <= 1:
            # 如果folds=1，使用单次划分 (80%训练, 20%测试)
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
            return list(sss.split(np.arange(len(labels)), labels))
            
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=123)
        return list(skf.split(np.arange(len(labels)), labels))
