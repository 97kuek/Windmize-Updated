import sys, os, csv, copy
import numpy as np        # preferred alias
import numpy              # keep legacy name access

# --- Qt5 imports ------------------------------------------------------------
from PyQt5 import QtCore, QtGui, QtWidgets
Qt = QtCore.Qt            # shorthand

# Give the most used Qt classes local aliases to keep the diff small
QApplication      = QtWidgets.QApplication
QMainWindow       = QtWidgets.QMainWindow
QWidget           = QtWidgets.QWidget
QTabWidget        = QtWidgets.QTabWidget
QGroupBox         = QtWidgets.QGroupBox
QFrame            = QtWidgets.QFrame
QLabel            = QtWidgets.QLabel
QLineEdit         = QtWidgets.QLineEdit
QPushButton       = QtWidgets.QPushButton
QCheckBox         = QtWidgets.QCheckBox
QProgressBar      = QtWidgets.QProgressBar
QTableWidget      = QtWidgets.QTableWidget
QTableWidgetItem  = QtWidgets.QTableWidgetItem
QHeaderView       = QtWidgets.QHeaderView
QSizePolicy       = QtWidgets.QSizePolicy
QHBoxLayout       = QtWidgets.QHBoxLayout
QVBoxLayout       = QtWidgets.QVBoxLayout
QFileDialog       = QtWidgets.QFileDialog
QMessageBox       = QtWidgets.QMessageBox
QSplashScreen     = QtWidgets.QSplashScreen
QPixmap           = QtGui.QPixmap
QIcon             = QtGui.QIcon

# --- Matplotlib Qt5 backend -------------------------------------------------
import matplotlib
matplotlib.use("Qt5Agg")   # ensure Qt 5 backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# ---------------------------------------------------------------------------


class Dataplot(FigureCanvas):
    """Reusable matplotlib canvas with helper ‘drawplot’ method."""
    def __init__(self, parent=None, width=8, height=3, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.tick_params(axis='both', which='major', labelsize=20)
        super().__init__(self.fig)                         # FigureCanvas init
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    # -----------------------------------------------------------------------
    def drawplot(self, x, y, x2=None, y2=None, *,
                 xlabel=None, ylabel=None, legend=None, aspect="equal"):

        self.axes.plot(x, y)
        if x2 is not None and y2 is not None:
            self.axes.plot(x2, y2, "--")

        if xlabel:
            self.axes.set_xlabel(xlabel, fontsize=20)
        if ylabel:
            self.axes.set_ylabel(ylabel, fontsize=20)
        if legend:
            self.axes.legend(legend, fontsize=15)
        if aspect == "auto":
            self.axes.set_aspect("auto")
        self.draw()


class ResultTabWidget(QTabWidget):
    """タブ切替でグラフ結果を表示するウィジェット"""
    def __init__(self, parent=None):
        super().__init__(parent)

        self.circulation_graph  = Dataplot()
        self.bending_graph      = Dataplot()
        self.bendingangle_graph = Dataplot()
        self.moment_graph       = Dataplot()
        self.shforce_graph      = Dataplot()
        self.ind_graph          = Dataplot()

        self.addTab(self.circulation_graph,  "循環分布")
        self.addTab(self.ind_graph,          "誘導角度[deg]")
        self.addTab(self.bending_graph,      "たわみ(軸:等倍)")
        self.addTab(self.bendingangle_graph, "たわみ角[deg]")
        self.addTab(self.moment_graph,       "曲げモーメント[N·m]")
        self.addTab(self.shforce_graph,      "せん断力[N]")


class ExeExportButton(QWidget):
    """計算実行・CSV出力・進捗バー"""
    def __init__(self, parent=None):
        super().__init__(parent)

        self.exebutton      = QPushButton("計算",       self)
        self.exportbutton   = QPushButton("CSV出力",    self)
        self.do_stracutual  = QCheckBox("構造考慮",     self)
        self.do_stracutual.setChecked(True)

        self.progressbar = QProgressBar(self)
        self.progressbar.setTextVisible(True)
        self.progressbar.setStyleSheet("""
            QProgressBar{
                border: 2px solid grey; border-radius: 5px; text-align: center;
            }
            QProgressBar::chunk{
                background-color: lightblue; width: 10px; margin: 1px;
            }
        """)

        lay = QHBoxLayout(self)
        lay.addStretch()
        lay.addWidget(self.progressbar)
        lay.addWidget(self.do_stracutual)
        lay.addWidget(self.exebutton)
        lay.addWidget(self.exportbutton)


class SettingWidget(QGroupBox):
    """設計パラメータ入力パネル（スパン分割テーブル等）"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("設計変数　※各翼終端位置は計算精度確保のため dy の整数倍とするとよい")

        font = QtGui.QFont()
        font.setPointSize(12)
        self.setFont(font)

        # --- 上段：基本入力欄 ------------------------------------------------
        self.lift_maxbending_input = QWidget(self)

        lbl = self.lift_maxbending_input
        lbl.liftlabel      = QLabel("揚力(kgf) :",        lbl)
        lbl.bendinglabel   = QLabel("  最大たわみ(mm) :", lbl)
        lbl.wireposlabel   = QLabel("  ワイヤー取付位置(mm) :", lbl)
        lbl.forcewirelabel = QLabel("  ワイヤー下向引張(N) :",  lbl)
        lbl.velocitylabel  = QLabel("  速度(m/s) :",      lbl)
        lbl.dylabel        = QLabel("  dy(mm) :",         lbl)

        def _le(width, text):
            le = QLineEdit(lbl)
            le.setFixedWidth(width)
            le.setText(text)
            return le

        lbl.liftinput     = _le(25,  "97")
        lbl.velocityinput = _le(33,  "7.21")
        lbl.bendinginput  = _le(33,  "2100")
        lbl.wireposinput  = _le(33,  "6250")
        lbl.forcewireinput= _le(25,  "485")
        lbl.dyinput       = _le(25,  "50")

        h1 = QHBoxLayout(lbl)
        h1.addStretch()
        for w in (lbl.liftlabel, lbl.liftinput,
                  lbl.velocitylabel, lbl.velocityinput,
                  lbl.bendinglabel, lbl.bendinginput,
                  lbl.wireposlabel, lbl.wireposinput,
                  lbl.forcewirelabel, lbl.forcewireinput,
                  lbl.dylabel, lbl.dyinput):
            h1.addWidget(w)
        lbl.setLayout(h1)

        # --- 中段：桁剛性設定ダイアログ呼び出し -----------------------------
        self.EIinput = QFrame(self)
        self.EIinput.EIinputbutton = QPushButton("桁詳細設定", self.EIinput)
        self.EIinput.EIinputbutton.setFixedWidth(100)
        h2 = QHBoxLayout(self.EIinput)
        h2.addStretch()
        h2.addWidget(self.EIinput.EIinputbutton)

        # --- スパン分割テーブル -------------------------------------------
        self.strechwid = QFrame(self)
        self.tablewidget = QTableWidget(self.strechwid)
        self.tablewidget.setMaximumSize(1000, 100)
        self.tablewidget.setMinimumSize(600, 100)
        self.tablewidget.setColumnCount(7)
        self.tablewidget.setRowCount(2)

        headers = ["", "第1翼", "第2翼", "第3翼", "第4翼", "第5翼", "第6翼"]
        for i, h in enumerate(headers):
            self.tablewidget.setHorizontalHeaderItem(i, QTableWidgetItem(h))

        self.tablewidget.setItem(0, 0, QTableWidgetItem("終端(mm)"))
        self.tablewidget.setItem(1, 0, QTableWidgetItem("調整係数"))
        for r in (0, 1):
            self.tablewidget.item(r, 0).setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

        span_list = [1100, 4300, 7500, 10200, 13150, 16250]
        for i in range(6):
            self.tablewidget.setItem(0, i + 1, QTableWidgetItem(str(span_list[i])))
            self.tablewidget.setItem(1, i + 1, QTableWidgetItem("1"))

        # stretch columns / rows
        self.tablewidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tablewidget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # テーブル下部の「列追加/削除」ボタン
        self.tablewidget.buttons = QWidget(self)
        self.tablewidget.insertcolumn = QPushButton("列追加", self.tablewidget.buttons)
        self.tablewidget.deletecolumn = QPushButton("列削除", self.tablewidget.buttons)
        hb = QHBoxLayout(self.tablewidget.buttons)
        hb.addStretch()
        hb.addWidget(self.tablewidget.insertcolumn)
        hb.addWidget(self.tablewidget.deletecolumn)

        # Frame layout
        hf = QHBoxLayout(self.strechwid)
        hf.addStretch()
        hf.addWidget(self.tablewidget)

        # --- 全体レイアウト ------------------------------------------------
        vlay = QVBoxLayout(self)
        vlay.addWidget(self.lift_maxbending_input)
        vlay.addWidget(self.tablewidget.buttons)
        vlay.addWidget(self.strechwid)
        vlay.addWidget(self.EIinput)
        self.setLayout(vlay)


class ResultValWidget(QGroupBox):
    """計算結果ラベルの束"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("計算結果")

        self.liftresultlabel  = QLabel("計算揚力[kgf] : --", self)
        self.Diresultlabel    = QLabel("   抗力[N] : --",      self)
        self.swresultlabel    = QLabel("   桁重量概算[kg] : --",self)
        self.lambda1label     = QLabel("   構造制約係数λ1[-] : --", self)
        self.lambda2label     = QLabel("   揚力制約係数λ2[-] : --", self)

        h = QHBoxLayout(self)
        h.addStretch()
        for w in (self.liftresultlabel, self.Diresultlabel,
                  self.swresultlabel, self.lambda1label, self.lambda2label):
            h.addWidget(w)


class EIsettingWidget(QtWidgets.QDialog):
    """桁剛性 / 線密度 入力用ダイアログ (複数タブ)"""
    def __init__(self, tablewidget, parent=None):
        super().__init__(parent)
        self.setFixedSize(600, 170)
        self.setModal(True)

        self.tabwidget = QTabWidget(self)
        v = QVBoxLayout(self)
        v.addWidget(self.tabwidget)

    # ---------------------------------------------------------------------
    def EIsetting(self, tablewidget):
        section_num = tablewidget.columnCount() - 1
        self.tabwidget.clear()

        self.EIinputWidget = []
        for i in range(section_num):
            gb = QGroupBox(f"第{i+1}翼の剛性と線密度を入力してください", self)
            table = QTableWidget(gb)
            table.setColumnCount(5)
            table.setRowCount(3)
            table.setFixedSize(570, 100)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

            for r, title in enumerate(("翼区切終端[mm]", "EI", "線密度[kg/m]")):
                table.setItem(r, 0, QTableWidgetItem(title))
                table.item(r, 0).setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            glay = QVBoxLayout(gb)
            glay.addWidget(table)
            self.tabwidget.addTab(gb, f"第{i+1}翼")

            # keep reference
            gb.EIinputtable = table
            self.EIinputWidget.append(gb)


# ---------------------------------------------------------------------------
# *** 計算エンジン本体 – 数値計算部は PyQt に非依存なのでオリジナルをほぼ維持 ***
# ---------------------------------------------------------------------------
class TR797_modified:
    """TR-797 最適循環分布 + 構造弾性計算ラッパ"""
    def __init__(self):
        # リスト変数定義（省略行は v---snip---v）
        self.dy = 0.05
        self.y_div = [];  self.z_div = [];  self.y_section = []; self.Ndiv_sec = []
        self.y = [];      self.z = [];      self.phi = []
        self.dS = []
        self.sigma = [];  self.spar_weight = 0; self.sigma_wire = []
        self.polize_mat = [[]]
        self.Q_ij = [[]]
        self.sh_mat = [[]]
        self.mo_mat = [[]]
        self.EI = [];     self.vd_mat = []; self.v_mat = []
        self.B = [[]];    self.C = [[]];    self.A = [[]]
        self.gamma = [];  self.ind_vel = []
        self.run = 1      # 外部 UI から停止要求を受けるフラグ
        self.comp = 1     # 計算完了フラグ

# ---------------------------------------------------------------------------
#  (続き)  TR797_modified – メソッド実装
# ---------------------------------------------------------------------------
    # -------------------------  入力値読み取り／前処理  ----------------------
    def prepare(self, settingwidget, eisettingwidget):
        self.dy         = float(settingwidget.lift_maxbending_input.dyinput.text()) / 1000
        self.b          = round(float(settingwidget.tablewidget.item(0,
                      settingwidget.tablewidget.columnCount() - 1).text()) * 2 / 1000, 4)
        self.n_section  = settingwidget.tablewidget.columnCount() - 1
        self.max_tawami = float(settingwidget.lift_maxbending_input.bendinginput.text()) / 1000
        self.y_wire     = float(settingwidget.lift_maxbending_input.wireposinput.text())  / 1000
        self.rho        = 1.154
        self.U          = float(settingwidget.lift_maxbending_input.velocityinput.text())
        self.M          = float(settingwidget.lift_maxbending_input.liftinput.text())

        # 区切り位置
        self.y_section.clear()
        for n in range(self.n_section):
            self.y_section.append(float(
                settingwidget.tablewidget.item(0, n + 1).text()) / 1000)

        # 1 区間 dy ステップで分割し、wire位置 / section index をメモ
        i = j = 0
        self.y_div.clear(); self.Ndiv_sec.clear()
        while True:
            self.y_div.append(round(self.dy * (i + 1), 4))

            if (round(self.y_div[i],4) > round(self.y_section[j]  - self.dy/2,4)
                    and round(self.y_div[i],4) <= round(self.y_section[j] + self.dy/2,4)):
                self.Ndiv_sec.append(i);  j += 1
            if (round(self.y_div[i],4) > round(self.y_wire - self.dy/2,4)
                    and round(self.y_div[i],4) <= round(self.y_wire + self.dy/2,4)):
                self.Ndiv_wire = i

            if j == self.n_section: break
            i += 1

        # パネル幅 dS と座標 (y,z,phi)
        self.dS.clear(); self.y.clear(); self.z.clear(); self.phi.clear(); self.z_div.clear()
        coe_tawami = self.max_tawami / (self.b/2)**2
        for n, yd in enumerate(self.y_div):
            self.z_div.append(coe_tawami * yd**2)
            if n:  # n != 0
                self.dS.append(np.sqrt((yd-self.y_div[n-1])**2 +
                                       (self.z_div[n]-self.z_div[n-1])**2) / 2)
                self.y.append((yd + self.y_div[n-1]) / 2)
                self.z.append((self.z_div[n] + self.z_div[n-1]) / 2)
                self.phi.append(np.arctan((self.z_div[n]-self.z_div[n-1]) /
                                           (yd - self.y_div[n-1])))
            else:
                self.dS.append(np.sqrt(yd**2 + self.z_div[n]**2) / 2)
                self.y.append(yd / 2)
                self.z.append(self.z_div[n] / 2)
                self.phi.append(np.arctan(self.z_div[n] / yd))

        # --- EI・線密度 σ の読み込み ------------------------------------
        self.EI.clear(); self.sigma.clear()
        n = 0
        for s in range(self.n_section):
            j = 1
            coe_EI = float(settingwidget.tablewidget.item(1, s+1).text())
            while True:
                if s == 0:
                    border = float(eisettingwidget.EIinputWidget[s].EIinputtable.item(0, j).text())/1000
                    cond   = self.y[n] < border
                else:
                    border = (float(eisettingwidget.EIinputWidget[s].EIinputtable.item(0, j).text())/1000
                              + self.y_section[s-1])
                    cond = self.y[n] < border
                idx = eisettingwidget.EIinputWidget[s].EIinputtable
                self.EI.append(float(idx.item(1, j).text()) * coe_EI)
                self.sigma.append(float(idx.item(2, j).text()) * coe_EI)
                if not cond: j += 1
                n += 1
                if n == len(self.y) or not self.y[n] < self.y_section[s]:
                    break

        # 桁重量
        self.spar_weight = np.sum(np.array(self.sigma) * np.array(self.dS) * 2)*2
        # 重力 + wire 張力 (下向き荷重)
        self.sigma_wire = copy.deepcopy(self.sigma)
        for i in range(len(self.y)):
            self.sigma_wire[i] *= 9.8
        self.sigma_wire[self.Ndiv_wire] += float(
            settingwidget.lift_maxbending_input.forcewireinput.text()
        ) / self.dS[self.Ndiv_wire] / 2

    # -----------------------------  行列生成  -------------------------------
    def matrix(self, progressbar, qApp):
        def calc_Q(y, z, phi, dS):
            """誘導速度係数行列 Q_ij を Bennet & Myers の式で評価"""
            N = len(y)
            Q_ij = np.zeros((N, N))
            for i in range(N):
                if self.run: break                       # stop requested
                for j in range(N):
                    qApp.processEvents()                # keep GUI responsive
                    progressbar.setValue(int((i*N + j + 1)/(N*N)*100))

                    # --- ベクトル差などをローカルで計算（速度優先） ---------
                    yd  =  (y[i]-y[j])*np.cos(phi[j]) + (z[i]-z[j])*np.sin(phi[j])
                    zd  = -(y[i]-y[j])*np.sin(phi[j]) + (z[i]-z[j])*np.cos(phi[j])
                    ydd =  (y[i]+y[j])*np.cos(phi[j]) - (z[i]-z[j])*np.sin(phi[j])
                    zdd =  (y[i]+y[j])*np.sin(phi[j]) + (z[i]-z[j])*np.cos(phi[j])

                    R2p = (yd-dS[j])**2 + zd**2
                    R2m = (yd+dS[j])**2 + zd**2
                    Rd2p= (ydd+dS[j])**2 + zdd**2
                    Rd2m= (ydd-dS[j])**2 + zdd**2

                    term1 = ((yd-dS[j])/R2p - (yd+dS[j])/R2m) * np.cos(phi[i]-phi[j])
                    term2 = (zd/R2p - zd/R2m)               * np.sin(phi[i]-phi[j])
                    term3 = ((ydd-dS[j])/Rd2m - (ydd+dS[j])/Rd2p) * np.cos(phi[i]+phi[j])
                    term4 = (zdd/Rd2m - zdd/Rd2p)               * np.sin(phi[i]+phi[j])

                    Q_ij[i,j] = -1/(2*np.pi) * (term1 + term2 + term3 + term4)
            return Q_ij

        # main part
        self.Q_ij = calc_Q(self.y, self.z, self.phi, self.dS)

        # --- 区分多角形化行列 P ------------------------------------------
        self.polize_mat = np.zeros((len(self.y), self.n_section))
        self.polize_mat[:self.Ndiv_sec[1]+1,0] = 1
        for j in range(1, self.n_section):
            idx0 = self.Ndiv_sec[j-1] + 1
            idx1 = self.Ndiv_sec[j] + 1
            y0, y1 = self.y_section[j-1], self.y_section[j]
            num = np.arange(idx0, idx1)
            self.polize_mat[num, j-1] = -(np.array(self.y)[num]-y1)/(y1-y0)
            self.polize_mat[num, j  ] =  (np.array(self.y)[num]-y0)/(y1-y0)

        # --- 積分行列（shear / moment / bending） ------------------------
        self.sh_mat = np.tril(np.ones((len(self.y), len(self.y)))) * 2
        self.sh_mat[np.diag_indices_from(self.sh_mat)] = 1
        self.sh_mat *= np.array(self.dS)[:,None] * self.U * self.rho

        self.mo_mat = np.tril(np.ones_like(self.sh_mat)) * 2
        self.mo_mat[np.diag_indices_from(self.mo_mat)] = 1
        self.mo_mat *= np.array(self.dS)[:,None]

        self.vd_mat = self.mo_mat / np.array(self.EI)[:,None] * 1e6
        self.v_mat  = self.mo_mat.copy()

        # --- 制約行列 B, C ----------------------------------------------
        B_want = np.zeros((1, len(self.y)));  B_want[0,-1] = 1
        self.B = B_want @ self.v_mat @ self.vd_mat @ self.mo_mat @ self.sh_mat @ self.polize_mat
        self.B_val = (self.max_tawami +
                      (B_want @ self.v_mat @ self.vd_mat @ self.mo_mat @ self.sh_mat)
                      @ (np.array(self.sigma_wire).T / self.rho / self.U))

        self.C     = 4 * self.rho * self.U * (np.array(self.dS) @ self.polize_mat)
        self.C_val = self.M * 9.8

    # ----------------------------  最適化 ----------------------------------
    def optimize(self, checkbox):
        """KKT 条件を線形方程式として解き、gamma_opt 等を更新"""
        do_structure = checkbox.isChecked()

        # Q_ij → induce-matrix A (symmetric)
        A = copy.deepcopy(self.Q_ij)
        for j in range(A.shape[1]):
            A[:,j] *= np.array(self.dS) * 2
        Mat_indD = A * self.rho                   # (for drag)
        A = (A + A.T)
        A = self.rho * self.polize_mat.T @ A @ self.polize_mat

        # 制約付加
                # --- constrain を付加する所だけ修正 --------------------------------
        if do_structure:          # 構造考慮あり
            A = np.vstack((A, -self.B, -self.C))

            col1 = np.append(-self.B, [0, 0])[:, None]   # (n_sec+2, 1)
            col2 = np.append(-self.C, [0, 0])[:, None]

            A = np.column_stack((A, col1, col2))

            rhs = np.zeros((A.shape[0], 1))
            rhs[-2, 0] = -float(self.B_val)      # ← .item() でも可
            rhs[-1, 0] = -float(self.C_val)
        else:                     # 構造無視
            A = np.vstack((A, -self.C))

            col = np.append(-self.C, [0])[:, None]       # (n_sec+1, 1)
            A  = np.column_stack((A, col))

            rhs = np.zeros((A.shape[0], 1))
            rhs[-1, 0] = -self.C_val


        sol             = np.linalg.solve(A, rhs)
        self.gamma_opt  = sol[:self.n_section]
        self.lambda1 = float(sol[self.n_section, 0]) if do_structure else 0.0
        self.lambda2 = float(sol[self.n_section + (1 if do_structure else 0), 0])


        # 各種派生量
        self.bending_mat   = self.v_mat @ self.vd_mat @ self.mo_mat @ self.sh_mat
        delta = self.polize_mat @ self.gamma_opt - np.array(self.sigma_wire)[:,None]/self.U/self.rho
        self.shearForce    = self.sh_mat @ delta
        self.moment        = self.mo_mat @ self.shearForce
        self.bending_angle = self.vd_mat @ self.moment
        self.bending       = self.bending_mat @ delta

        # 楕円揚力分布
        root_gamma = self.M*9.8 * 4 / np.pi / self.b / self.U / self.rho
        self.gamma_el = np.sqrt(np.maximum(root_gamma**2 -
                                           (np.array(self.y)*root_gamma/self.b*2)**2, 0))

        self.gamma    = self.polize_mat @ self.gamma_opt
        self.ind_vel  = (self.Q_ij / 2) @ self.gamma
        self.Di       = float(np.sum(self.rho * self.ind_vel * self.gamma * self.dy * 2))
        self.Lift = (self.C @ self.gamma_opt).item()
        self.comp     = 0         # 計算完了フラグ


# ---------------------------------------------------------------------------
#                                 GUI エントリ
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("WM.ico"))

    # Splash
    splash = QSplashScreen(QPixmap("WM_splash.png"))
    splash.show()

    # Widgets
    mainwindow        = QMainWindow()
    mainwindow.setWindowTitle("Windmize")
    resulttabwidget   = ResultTabWidget()
    exeexportbutton   = ExeExportButton()
    settingwidget     = SettingWidget()
    resultvalwidget   = ResultValWidget()
    eisettingwidget   = EIsettingWidget(settingwidget.tablewidget)
    eisettingwidget.EIsetting(settingwidget.tablewidget)

    # --- EI サンプル値を投入（旧ロジックそのまま） ------------------------
    EI_samples     = [3.4375e10, 3.6671e10, 1.6774e10, 8.3058e9, 1.8648e9, 7.094e7]
    sigma_samples  = [0.377, 0.357, 0.284, 0.245, 0.0929, 0.0440]
    for i_wing in range(settingwidget.tablewidget.columnCount()-1):
        for s in range(4):
            tbl = eisettingwidget.EIinputWidget[i_wing].EIinputtable
            tbl.setItem(1, s+1, QTableWidgetItem(str(EI_samples[i_wing])))
            tbl.setItem(2, s+1, QTableWidgetItem(str(sigma_samples[i_wing])))

    TR797_opt = TR797_modified()

    # ----------------------  コールバック定義 -----------------------------
    def insertcolumn():
        tw = settingwidget.tablewidget
        col = tw.columnCount()
        tw.setColumnCount(col+1)
        tw.setHorizontalHeaderItem(col, QTableWidgetItem(f"第{col}翼"))
        tw.setItem(0, col, QTableWidgetItem(str(
            float(tw.item(0, col-1).text()) + 2000)))
        tw.setItem(1, col, QTableWidgetItem("1"))
        tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # EIタブ追加
        i = col-1
        eisettingwidget.EIinputWidget.append(QGroupBox(f"第{i+1}翼の剛性と線密度を入力してください"))
        gb = eisettingwidget.EIinputWidget[i]
        table = QTableWidget(3,5, gb);  table.setFixedSize(570,100)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for r, t in enumerate(("翼区切終端[mm]","EI[N·mm^2]","線密度[kg/m]")):
            table.setItem(r,0, QTableWidgetItem(t))
            table.item(r,0).setFlags(Qt.ItemIsSelectable|Qt.ItemIsEnabled)
        gb.EIinputtable = table
        gl = QVBoxLayout(gb); gl.addWidget(table)
        eisettingwidget.tabwidget.addTab(gb, f"第{i+1}翼")

    def deletecolumn():
        tw = settingwidget.tablewidget
        col = tw.columnCount()
        if col >= 3:
            tw.setColumnCount(col-1)
            tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            eisettingwidget.tabwidget.removeTab(col-2)
            eisettingwidget.EIinputWidget.pop(col-2)

    def EIsettingshow():
        init_EI_widget()
        eisettingwidget.show()

    def init_EI_widget():
        """列数・終端位置に合わせて EI ダイアログをリフレッシュ"""
        y_div = [float(settingwidget.tablewidget.item(0,i+1).text())
                 for i in range(settingwidget.tablewidget.columnCount()-1)]
        for i, gb in enumerate(eisettingwidget.EIinputWidget):
            if i == 0:
                base = y_div[i]
            else:
                base = y_div[i] - y_div[i-1]
            pos = [base/4, base/2, base*3/4, base]
            for s in range(4):
                itm = QTableWidgetItem(str(pos[s]))
                itm.setFlags(Qt.ItemIsSelectable|Qt.ItemIsEnabled)
                gb.EIinputtable.setItem(0, s+1, itm)

    def show_results():
        # グラフ
        resulttabwidget.circulation_graph.axes.clear()
        resulttabwidget.circulation_graph.drawplot(
            np.array(TR797_opt.y).T,
            TR797_opt.gamma,
            np.array(TR797_opt.y),
            TR797_opt.gamma_el,
            xlabel="y[m]", ylabel="gamma[m^2/s]",
            legend=("optimized","elliptical"), aspect="auto")

        resulttabwidget.bending_graph.axes.clear()
        resulttabwidget.bending_graph.drawplot(
            np.array(TR797_opt.y), TR797_opt.bending,
            xlabel="y[m]", ylabel="bending[m]")

        resulttabwidget.ind_graph.axes.clear()
        resulttabwidget.ind_graph.drawplot(
            np.array(TR797_opt.y),
            np.degrees(np.arctan(-TR797_opt.ind_vel / TR797_opt.U)),
            xlabel="y[m]", ylabel="induced angle[deg]", aspect="auto")

        resulttabwidget.bendingangle_graph.axes.clear()
        resulttabwidget.bendingangle_graph.drawplot(
            np.array(TR797_opt.y),
            np.degrees(TR797_opt.bending_angle),
            xlabel="y[m]", ylabel="bending angle[deg]")

        resulttabwidget.moment_graph.axes.clear()
        resulttabwidget.moment_graph.drawplot(
            np.array(TR797_opt.y),
            TR797_opt.moment,
            xlabel="y[m]", ylabel="moment[Nm]")

        resulttabwidget.shforce_graph.axes.clear()
        resulttabwidget.shforce_graph.drawplot(
            np.array(TR797_opt.y),
            TR797_opt.shearForce,
            xlabel="y[m]", ylabel="shearforce[N]")

        # 数値ラベル
        resultvalwidget.liftresultlabel.setText(f"計算揚力[kgf] : {TR797_opt.Lift/9.8:.3f}")
        resultvalwidget.Diresultlabel.setText(  f"   抗力[N] : {TR797_opt.Di:.3f}")
        resultvalwidget.swresultlabel.setText(  f"   桁重量概算[kg] : {TR797_opt.spar_weight:.3f}")
        if exeexportbutton.do_stracutual.isChecked():
            resultvalwidget.lambda1label.setText(f"   構造制約係数λ1[-] : {TR797_opt.lambda1:.3f}")
        else:
            resultvalwidget.lambda1label.setText("   構造制約係数λ1[-] : --")
        resultvalwidget.lambda2label.setText(  f"   揚力制約係数λ2[-] : {TR797_opt.lambda2:.3f}")

    def calculation():
        if TR797_opt.run:                                # idle → start
            exeexportbutton.exebutton.setText("計算中止")
            init_EI_widget()
            TR797_opt.__init__()                         # reset
            TR797_opt.prepare(settingwidget, eisettingwidget)
            TR797_opt.run = 0
            TR797_opt.matrix(exeexportbutton.progressbar, app)
            if not TR797_opt.run:
                TR797_opt.optimize(exeexportbutton.do_stracutual)
                show_results()
            TR797_opt.run = 1
            exeexportbutton.exebutton.setText("計算")
        else:                                            # running → stop
            TR797_opt.run = 1
            exeexportbutton.exebutton.setText("計算")

    def exportCSV():
        if TR797_opt.comp:
            QMessageBox.warning(mainwindow,"Error","計算が完了していません")
            return
        fname, _ = QFileDialog.getSaveFileName(mainwindow, "計算結果出力", "", "CSV (*.csv)")
        if not fname: return
        try:
            with open(fname, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["循環分布最化結果"])
                w.writerow(["揚力[kgf]",      round(TR797_opt.Lift/9.8,3)])
                w.writerow(["抗力[N]",        round(TR797_opt.Di,3)])
                w.writerow(["桁重量概算[kg]", round(TR797_opt.spar_weight,3)])
                w.writerow(["構造制約係数[-]",
                            "--" if not exeexportbutton.do_stracutual.isChecked()
                            else round(TR797_opt.lambda1,3)])
                w.writerow(["揚力制約係数[-]", round(TR797_opt.lambda2,3)])
                w.writerow([])
                w.writerow(["以下 翼セクションでの値"])
                w.writerow(["y[m]", "gamma[m^2/s]"])
                buf = np.hstack([np.array([TR797_opt.y_section]).T, TR797_opt.gamma_opt])
                for row in buf: w.writerow(row)
                w.writerow([])
                w.writerow(["以下 Control Point での値"])
                w.writerow(["y[m]","gamma","ind[deg]","bending[m]","bendAng[deg]",
                            "moment","shear","EI","sigma"])
                for i, y in enumerate(TR797_opt.y):
                    w.writerow([y,
                                TR797_opt.gamma[i,0],
                                np.degrees(np.arctan(TR797_opt.ind_vel[i,0]/TR797_opt.U)),
                                TR797_opt.bending[i,0],
                                np.degrees(TR797_opt.bending_angle[i,0]),
                                TR797_opt.moment[i,0],
                                TR797_opt.shearForce[i,0],
                                TR797_opt.EI[i],
                                TR797_opt.sigma[i]])
        except Exception as e:
            QMessageBox.warning(mainwindow, "Error", f"CSV出力に失敗しました\n{e}")

    # ----------------------  メニュー / シグナル配線 -----------------------
    menubar   = mainwindow.menuBar()
    filemenu  = menubar.addMenu("File")
    filemenu.addAction("ファイルI/O未実装")

    aboutmenu = menubar.addMenu("About")
    act_about = aboutmenu.addAction("About Windmize")
    act_about.triggered.connect(lambda: QMessageBox.about(
        mainwindow, "About Windmize",
        "<h2>Windmize 1.00</h2>"
        "<p>Copyright (C) 2014 Naoto Morita</p>"
        "<p>Windmize is without any warranty. "
        "This program has been developed excusively for the design of aerowings."
        "Any other usage is strongly disapproved.</p>"
        "<p>Distributed under the GNU GPL.</p>"))

    # signal wiring
    settingwidget.tablewidget.insertcolumn.clicked.connect(insertcolumn)
    settingwidget.tablewidget.deletecolumn.clicked.connect(deletecolumn)
    settingwidget.EIinput.EIinputbutton.clicked.connect(EIsettingshow)
    exeexportbutton.exebutton.clicked.connect(calculation)
    exeexportbutton.exportbutton.clicked.connect(exportCSV)

    # --- メインレイアウト --------------------------------------------------
    mainpanel = QWidget()
    vlay = QVBoxLayout(mainpanel)
    vlay.addWidget(resulttabwidget)
    vlay.addWidget(resultvalwidget)
    vlay.addWidget(exeexportbutton)
    vlay.addWidget(settingwidget)
    mainwindow.setCentralWidget(mainpanel)

    mainwindow.resize(1200, 900)
    mainwindow.show()
    splash.finish(mainwindow)
    sys.exit(app.exec_())


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()