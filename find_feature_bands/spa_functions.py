import numpy as np
from scipy.linalg import qr
import scipy.stats
from progress.bar import Bar
from matplotlib import pyplot as plt
import matplotlib


class SPA:
    def _projections_qr(self, X, k, M):
        '''
        X : 预测变量矩阵
        K ：投影操作的初始列的索引
        M : 结果包含的变量个数
        return ：由投影操作生成的变量集的索引
        '''
        X_projected = X.copy()
        # 计算列向量的平方和
        norms = np.sum((X ** 2), axis=0)
        # 找到norms中数值最大列的平方和
        norm_max = np.amax(norms)
        # 缩放第K列 使其成为“最大的”列
        X_projected[:, k] = X_projected[:, k] * 2 * norm_max / norms[k]
        # 矩阵分割 ，order 为列交换索引
        _, __, order = qr(X_projected, 0, pivoting=True)
        return order[:M].T

    def _validation(self, Xcal, ycal, var_sel, Xval=None, yval=None):
        '''
        [yhat,e] = validation(Xcal,var_sel,ycal,Xval,yval) -->  使用单独的验证集进行验证
        [yhat,e] = validation(Xcal,ycalvar_sel) --> 交叉验证
        '''
        N = Xcal.shape[0]  # N 测试集的个数
        if Xval is None:  # 判断是否使用验证集
            NV = 0
        else:
            NV = Xval.shape[0]  # NV 验证集的个数

        yhat = e = None
        # 使用单独的验证集进行验证
        if NV > 0:
            Xcal_ones = np.hstack(
                [np.ones((N, 1)), Xcal[:, var_sel].reshape(N, -1)])
            # 对偏移量进行多元线性回归
            b = np.linalg.lstsq(Xcal_ones, ycal, rcond=None)[0]
            # 对验证集进行预测
            np_ones = np.ones((NV, 1))
            Xval_ = Xval[:, var_sel]
            X = np.hstack([np.ones((NV, 1)), Xval[:, var_sel]])
            yhat = X.dot(b)

            # jicm
            # threshold_value = 0.5
            # yhat = np.array([0 if value_ < threshold_value else 1 for value_ in yhat])

            # 计算误差
            e = yval - yhat
        else:
            # 为yhat 设置适当大小
            yhat = np.zeros((N, 1))
            for i in range(N):
                # 从测试集中 去除掉第 i 项
                cal = np.hstack([np.arange(i), np.arange(i + 1, N)])
                X = Xcal[cal, var_sel.astype(np.int)]
                y = ycal[cal]
                xtest = Xcal[i, var_sel]
                # ytest = ycal[i]
                X_ones = np.hstack([np.ones((N - 1, 1)), X.reshape(N - 1, -1)])
                # 对偏移量进行多元线性回归
                b = np.linalg.lstsq(X_ones, y, rcond=None)[0]
                # 对验证集进行预测
                yhat_ = np.hstack([np.ones(1), xtest]).dot(b)

                # jicm
                # threshold_value = 0.5
                # yhat_ = np.array([0 if value_ < threshold_value else 1 for value_ in yhat_])
                yhat[i] = yhat_

            # 计算误差
            e = ycal - yhat

        return yhat, e

    def spa(self, Xcal, ycal, m_min=1, m_max=None, Xval=None, yval=None, autoscaling=1):
        '''
        [var_sel,var_sel_phase2] = spa(Xcal,ycal,m_min,m_max,Xval,yval,autoscaling) --> 使用单独的验证集进行验证
        [var_sel,var_sel_phase2] = spa(Xcal,ycal,m_min,m_max,autoscaling) --> 交叉验证

        如果 m_min 为空时， 默认 m_min = 1
        如果 m_max 为空时：
            1. 当使用单独的验证集进行验证时， m_max = min(N-1, K)
            2. 当使用交叉验证时, m_max = min(N-2, K)
        autoscaling : 是否使用自动刻度 yes = 1, no = 0, 默认为 1
        '''
        assert (autoscaling == 0 or autoscaling == 1), "请选择是否使用自动计算"
        N, K = Xcal.shape
        if m_max is None:
            if Xval is None:
                m_max = min(N - 1, K)
            else:
                m_max = min(N - 2, K)
        assert (m_max < min(N - 1, K)), "m_max 参数异常"

        # 第一步： 对测试集进行投影操作
        # 在均值中心化 和 自动窗口 之后 对 Xcal的列进行投影操作
        normalization_factor = None
        if autoscaling == 1:
            normalization_factor = np.std(
                Xcal, ddof=1, axis=0).reshape(1, -1)[0]
        else:
            normalization_factor = np.ones((1, K))[0]

        Xcaln = np.empty((N, K))
        for k in range(K):
            x = Xcal[:, k]
            Xcaln[:, k] = (x - np.mean(x)) / normalization_factor[k]

        SEL = np.zeros((m_max, K))
        # 进度条
        with Bar('Projections :', max=K) as bar:
            for k in range(K):
                SEL[:, k] = self._projections_qr(Xcaln, k, m_max)
                bar.next()

        # 第二步： 进行评估  loss
        PRESS = float('inf') * np.ones((m_max + 1, K))
        with Bar('Evaluation of variable subsets :', max=(K) * (m_max - m_min + 1)) as bar:
            for k in range(K):
                for m in range(m_min, m_max + 1):
                    var_sel = SEL[:m, k].astype(np.int)
                    _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)
                    PRESS[m, k] = e.T.dot(e)
                    bar.next()
        PRESSmin = np.min(PRESS, axis=0)
        m_sel = np.argmin(PRESS, axis=0)
        k_sel = np.argmin(PRESSmin)
        # 第 k_sel 波段为初始波段时最佳，波段数目为 m_sel(k_sel)
        var_sel_phase2 = SEL[:m_sel[k_sel], k_sel].astype(np.int)

        # 最后消去变量
        # 第 3.1 步 计算相关指数
        Xcal2 = np.hstack([np.ones((N, 1)), Xcal[:, var_sel_phase2]])
        b = np.linalg.lstsq(Xcal2, ycal, rcond=None)[0]
        # 每一列取标准差值 消除样本内部的误差
        std_deviation = np.std(Xcal2, ddof=1, axis=0)

        relev = np.abs(b * std_deviation.T)
        relev = relev[1:]  # TODO 选取后的波长乘以标准差之后的值

        index_increasing_relev = np.argsort(relev, axis=0)
        index_decreasing_relev = index_increasing_relev[::-1].reshape(1, -1)[0]

        PRESS_scree = np.empty(len(var_sel_phase2))
        yhat = e = None
        for i in range(len(var_sel_phase2)):
            var_sel = var_sel_phase2[index_decreasing_relev[:i + 1]]
            _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)

            PRESS_scree[i] = np.conj(e).T.dot(e)

        RMSEP_scree = np.sqrt(PRESS_scree / len(e))

        # 第 3.3： F-test 验证
        PRESS_scree_min = np.min(PRESS_scree)
        alpha = 0.25
        dof = len(e)
        fcrit = scipy.stats.f.ppf(1 - alpha, dof, dof)
        PRESS_crit = PRESS_scree_min * fcrit
        print('PRESS_scree: ', PRESS_scree)
        print('PRESS_crit: ', PRESS_crit)

        # 找到不明显比 PRESS_scree_min 大的最小变量
        i_crit = np.min(np.nonzero(PRESS_scree < PRESS_crit))  # 显著的最少个数
        print('i_crit: ', i_crit)

        i_crit = max(m_min, i_crit)  # 和设定的最小个数进行对比

        # todo 最终选定的 var_sel 候选的: var_sel_phase2[index_decreasing_relev] 对应的数值
        # todo
        # todo 二阶段的趋势曲线: PRESSmin(误差内积) 选定k_sel 作为初始变量  PRESS[k_sel]  0-1为inf
        # todo 三阶段的趋势曲线: PRESS_scree(误差) 去除不显著的剩余最少数目 i_crit
        # todo 三阶段的相关性系数 relev(表格)   relev[index_decreasing_relev][:i_crit] 从大到小
        var_sel = var_sel_phase2[index_decreasing_relev[:i_crit]]

        colors = ['#EA8379', '#7DAEE0', '#FF9896', '#DBDB8D', '#C59D94', 'tab:cyan']

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'
        matplotlib.rc("font", family='MicroSoft YaHei', weight="bold")

        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        fig1 = plt.figure()
        plt.xlabel('$Number$ $of$ $wavelengths$ $included$', fontsize=12)
        plt.ylabel('$RMSE$', fontsize=12)

        print('length:', len(var_sel), 'i_crit: ', i_crit)
        print('Final number of selected variables:{}(RMSE={})'.format(len(var_sel), RMSEP_scree[i_crit-1]))
        plt.plot(RMSEP_scree)
        plt.scatter(i_crit, RMSEP_scree[i_crit-1], marker='s', color=colors[0], s=18, label='finally selected number')  # finally

        fig2 = plt.figure()
        plt.plot(Xcal[0, :], label='first spectrum')
        plt.scatter(var_sel, Xcal[0, var_sel], marker='s', color='r', label='selected wavelength', s=18)  # finally
        x_candidate = np.sort(var_sel_phase2)
        x_candidate_without_final = np.setdiff1d(x_candidate, var_sel)
        plt.scatter(x_candidate_without_final, Xcal[0, x_candidate_without_final], marker='o', color=colors[1], edgecolors='gray', label='candidate wavelength', s=18)  # finally

        plt.legend()
        plt.xlabel('$Wavelength$ $index$', fontsize=12)
        plt.ylabel('$Transmission$', fontsize=12)
        plt.show()
        print('Sorted selected wavelength:', var_sel)
        print('Sorted selected relevance:', relev[index_decreasing_relev][:i_crit])
        print('Sorted candidate wavelength:', var_sel_phase2[index_decreasing_relev])
        print('Sorted candidate relevance:', relev[index_decreasing_relev][:len(var_sel_phase2)])
        print('following RMSEP_scree:', RMSEP_scree)
        print('best nums:', i_crit)
        # print('Spectrum:', Xcal[0])

        return var_sel, var_sel_phase2, relev, index_decreasing_relev, RMSEP_scree

    def __repr__(self):
        return "SPA()"




