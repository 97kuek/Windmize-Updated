# ============================================================
#  windmize.py   (PyQt5)               ← Part 1 / 3
# ============================================================

import sys, os, csv, copy, json
import numpy as np
import numpy                          # 旧名互換

# ---- Qt5 ---------------------------------------------------
from PyQt5 import QtCore, QtGui, QtWidgets
Qt = QtCore.Qt
QApplication   = QtWidgets.QApplication
QMainWindow    = QtWidgets.QMainWindow
QWidget        = QtWidgets.QWidget
QTabWidget     = QtWidgets.QTabWidget
QGroupBox      = QtWidgets.QGroupBox
QFrame         = QtWidgets.QFrame
QLabel         = QtWidgets.QLabel
QLineEdit      = QtWidgets.QLineEdit
QPushButton    = QtWidgets.QPushButton
QCheckBox      = QtWidgets.QCheckBox
QProgressBar   = QtWidgets.QProgressBar
QTableWidget   = QtWidgets.QTableWidget
QTableWidgetItem = QtWidgets.QTableWidgetItem
QHeaderView    = QtWidgets.QHeaderView
QHBoxLayout    = QtWidgets.QHBoxLayout
QVBoxLayout    = QtWidgets.QVBoxLayout
QFileDialog    = QtWidgets.QFileDialog
QMessageBox    = QtWidgets.QMessageBox
QSplashScreen  = QtWidgets.QSplashScreen
QPixmap        = QtGui.QPixmap
QIcon          = QtGui.QIcon

# ---- matplotlib -------------------------------------------
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# -----------------------------------------------------------


# ---------- 共通の小さな描画キャンバス ----------------------
class Dataplot(FigureCanvas):
    def __init__(self, parent=None, width=8, height=3, dpi=80):
        self.fig  = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.tick_params(axis="both", which="major", labelsize=20)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()

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


# ---------- グラフタブ --------------------------------------
class ResultTabWidget(QTabWidget):
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


# ---------- 実行／出力ボタン＋プログレスバー -----------------
class ExeExportButton(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.exebutton     = QPushButton("計算",    self)
        self.exportbutton  = QPushButton("CSV出力", self)
        self.do_stracutual = QCheckBox("構造考慮",  self)
        self.do_stracutual.setChecked(True)

        self.progressbar = QProgressBar(self)
        self.progressbar.setTextVisible(True)
        self.progressbar.setStyleSheet("""
            QProgressBar{border:2px solid grey;border-radius:5px;text-align:center;}
            QProgressBar::chunk{background-color:lightblue;width:10px;margin:1px;}
        """)

        lay = QHBoxLayout(self)
        lay.addStretch(1)
        for w in (self.progressbar, self.do_stracutual,
                  self.exebutton, self.exportbutton):
            lay.addWidget(w)


# ---------- 入力パネル（長いので省略せずそのまま） -------------
#   ※途中まで同一、最後のレイアウト作成で終了
class SettingWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("設計変数　※各翼終端位置は dy の整数倍推奨")
        font = QtGui.QFont(); font.setPointSize(12); self.setFont(font)

        # ---- 上段入力 ---------------------------------------
        self.lift_maxbending_input = QWidget(self)
        lbl = self.lift_maxbending_input
        lbl.liftlabel      = QLabel("揚力(kgf) :", lbl)
        lbl.bendinglabel   = QLabel("  最大たわみ(mm) :", lbl)
        lbl.wireposlabel   = QLabel("  ワイヤー取付位置(mm) :", lbl)
        lbl.forcewirelabel = QLabel("  ワイヤー下向引張(N) :", lbl)
        lbl.velocitylabel  = QLabel("  速度(m/s) :", lbl)
        lbl.dylabel        = QLabel("  dy(mm) :", lbl)

        def _le(w, txt): le = QLineEdit(lbl); le.setFixedWidth(w); le.setText(txt); return le
        lbl.liftinput      = _le(25, "97")
        lbl.velocityinput  = _le(33, "7.2")
        lbl.bendinginput   = _le(33, "2100")
        lbl.wireposinput   = _le(33, "6250")
        lbl.forcewireinput = _le(25, "485")
        lbl.dyinput        = _le(25, "50")

        h1 = QHBoxLayout(lbl); h1.addStretch()
        for w in (lbl.liftlabel, lbl.liftinput,
                  lbl.velocitylabel, lbl.velocityinput,
                  lbl.bendinglabel, lbl.bendinginput,
                  lbl.wireposlabel, lbl.wireposinput,
                  lbl.forcewirelabel, lbl.forcewireinput,
                  lbl.dylabel, lbl.dyinput):
            h1.addWidget(w)
        lbl.setLayout(h1)

        # ---- 中段：桁剛性ダイアログ ----
        self.EIinput = QFrame(self)
        self.EIinput.EIinputbutton = QPushButton("桁詳細設定", self.EIinput)
        h2 = QHBoxLayout(self.EIinput); h2.addStretch(); h2.addWidget(self.EIinput.EIinputbutton)

        # ---- スパン分割テーブル ----
        self.strechwid    = QFrame(self)
        self.tablewidget  = QTableWidget(self.strechwid)
        self.tablewidget.setMaximumSize(1000,100)
        self.tablewidget.setMinimumSize(600,100)
        self.tablewidget.setColumnCount(7); self.tablewidget.setRowCount(2)
        cols = ["","第1翼","第2翼","第3翼","第4翼","第5翼","第6翼"]
        for i,c in enumerate(cols): self.tablewidget.setHorizontalHeaderItem(i,QTableWidgetItem(c))
        self.tablewidget.setItem(0,0,QTableWidgetItem("終端(mm)"))
        self.tablewidget.setItem(1,0,QTableWidgetItem("調整係数"))
        for r in (0,1):
            self.tablewidget.item(r,0).setFlags(Qt.ItemIsSelectable|Qt.ItemIsEnabled)
        span_list=[1100,4300,7500,10200,13150,16250]
        for i,s in enumerate(span_list):
            self.tablewidget.setItem(0,i+1,QTableWidgetItem(str(s)))
            self.tablewidget.setItem(1,i+1,QTableWidgetItem("1"))
        self.tablewidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tablewidget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 追加/削除ボタン
        self.tablewidget.buttons   = QWidget(self)
        self.tablewidget.insertcolumn = QPushButton("列追加", self.tablewidget.buttons)
        self.tablewidget.deletecolumn = QPushButton("列削除", self.tablewidget.buttons)
        hb = QHBoxLayout(self.tablewidget.buttons); hb.addStretch()
        hb.addWidget(self.tablewidget.insertcolumn); hb.addWidget(self.tablewidget.deletecolumn)

        hf = QHBoxLayout(self.strechwid); hf.addStretch(); hf.addWidget(self.tablewidget)

        # ---- 全体レイアウト ----
        vlay = QVBoxLayout(self)
        vlay.addWidget(self.lift_maxbending_input)
        vlay.addWidget(self.tablewidget.buttons)
        vlay.addWidget(self.strechwid)
        vlay.addWidget(self.EIinput)
# ============================================================
#  windmize.py   (PyQt5)               ← Part 2 / 3
# ============================================================

# ------- 計算結果ラベル -------------------------------------
class ResultValWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("計算結果")
        self.liftresultlabel  = QLabel("計算揚力[kgf] : --", self)
        self.Diresultlabel    = QLabel("   抗力[N] : --",    self)
        self.swresultlabel    = QLabel("   桁重量概算[kg] : --", self)
        self.lambda1label     = QLabel("   構造制約係数λ1[-] : --", self)
        self.lambda2label     = QLabel("   揚力制約係数λ2[-] : --", self)

        h = QHBoxLayout(self); h.addStretch(1)
        for w in (self.liftresultlabel, self.Diresultlabel,
                  self.swresultlabel, self.lambda1label, self.lambda2label):
            h.addWidget(w)


# ------- EI 詳細入力ダイアログ --------------------------------
class EIsettingWidget(QtWidgets.QDialog):
    def __init__(self, tablewidget, parent=None):
        super().__init__(parent)
        self.setFixedSize(600, 170); self.setModal(True)
        self.tabwidget = QTabWidget(self)
        v = QVBoxLayout(self); v.addWidget(self.tabwidget)

    # 呼び出し時に列数を見てタブ再作成
    def EIsetting(self, tablewidget):
        self.tabwidget.clear()
        self.EIinputWidget = []
        for i in range(tablewidget.columnCount()-1):
            gb = QGroupBox(f"第{i+1}翼の剛性と線密度を入力してください", self)
            tbl = QTableWidget(3,5, gb); tbl.setFixedSize(570,100)
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tbl.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
            for r, t in enumerate(("翼区切終端[mm]","EI","線密度[kg/m]")):
                tbl.setItem(r,0,QTableWidgetItem(t))
                tbl.item(r,0).setFlags(Qt.ItemIsSelectable|Qt.ItemIsEnabled)
            gb.EIinputtable = tbl
            gl = QVBoxLayout(gb); gl.addWidget(tbl)
            self.tabwidget.addTab(gb, f"第{i+1}翼")
            self.EIinputWidget.append(gb)


# ============================================================
#      ↓↓↓   数値計算エンジン（オリジナルを簡潔に）   ↓↓↓
# ============================================================
class TR797_modified:
    def __init__(self):
        self.dy = 0.05
        self.y_div = []; self.z_div = []; self.y_section=[]; self.Ndiv_sec=[]
        self.y=[]; self.z=[]; self.phi=[]; self.dS=[]
        self.sigma=[]; self.spar_weight=0; self.sigma_wire=[]
        self.polize_mat=[[]]; self.Q_ij=[[]]; self.sh_mat=[[]]; self.mo_mat=[[]]
        self.EI=[]; self.vd_mat=[]; self.v_mat=[]
        self.B=[[]]; self.C=[[]]
        self.gamma=[]; self.ind_vel=[]
        self.run  = 1   # 1=idle / 0=calculating
        self.comp = 1   # 0=結果あり

    # ------------- prepare() : 入力取り込み & メッシュ生成 ---------------
    def prepare(self, settingwidget, eisettingwidget):
        lbl  = settingwidget.lift_maxbending_input
        self.dy   = float(lbl.dyinput.text())/1000
        self.max_tawami = float(lbl.bendinginput.text())/1000
        self.y_wire     = float(lbl.wireposinput.text())/1000
        self.U   = float(lbl.velocityinput.text())
        self.M   = float(lbl.liftinput.text())
        self.rho = 1.154

        # セクション境界
        tw = settingwidget.tablewidget
        self.n_section = tw.columnCount()-1
        self.y_section = [float(tw.item(0,i+1).text())/1000 for i in range(self.n_section)]
        self.b = round(float(tw.item(0, self.n_section).text())*2/1000,4)

        # スパン方向分割
        self.y_div=[]; self.Ndiv_sec=[]
        i=j=0
        while True:
            self.y_div.append(round(self.dy*(i+1),4))
            if abs(self.y_div[i]-self.y_section[j])<=self.dy/2:
                self.Ndiv_sec.append(i); j+=1
            if abs(self.y_div[i]-self.y_wire)<=self.dy/2:
                self.Ndiv_wire=i
            if j==self.n_section: break
            i+=1

        # パネル中央点 y,z,phi と dS
        self.y=[]; self.z=[]; self.phi=[]; self.dS=[]; self.z_div=[]
        coef = self.max_tawami/(self.b/2)**2
        for n,yd in enumerate(self.y_div):
            self.z_div.append(coef*yd**2)
            if n:
                self.dS.append(np.hypot(yd-self.y_div[n-1],
                                        self.z_div[n]-self.z_div[n-1])/2)
                self.y.append((yd+self.y_div[n-1])/2)
                self.z.append((self.z_div[n]+self.z_div[n-1])/2)
                self.phi.append(np.arctan((self.z_div[n]-self.z_div[n-1])/
                                          (yd-self.y_div[n-1])))
            else:
                self.dS.append(np.hypot(yd,self.z_div[n])/2)
                self.y.append(yd/2); self.z.append(self.z_div[n]/2)
                self.phi.append(np.arctan(self.z_div[n]/yd))

        # EI, σ 読み込み
        self.EI=[]; self.sigma=[]
        p=0
        for s in range(self.n_section):
            tbl = eisettingwidget.EIinputWidget[s].EIinputtable
            coefEI = float(tw.item(1,s+1).text())
            j=1
            while p<len(self.y) and self.y[p]<self.y_section[s]:
                border = float(tbl.item(0,j).text())/1000
                if s>0: border += self.y_section[s-1]
                self.EI.append(float(tbl.item(1,j).text())*coefEI)
                self.sigma.append(float(tbl.item(2,j).text())*coefEI)
                if self.y[p]>=border: j+=1
                p+=1

        self.spar_weight = np.sum(np.array(self.sigma)*np.array(self.dS)*4)
        self.sigma_wire  = [s*9.8 for s in self.sigma]
        self.sigma_wire[self.Ndiv_wire]+=float(lbl.forcewireinput.text())/self.dS[self.Ndiv_wire]/2

    # ------------- 以下 matrix(), optimize() は変更なし (省略不可の方は元コードを使用) --------------
# ============================================================
#  windmize.py   (PyQt5)               ← Part 3 / 3
# ============================================================

    # ------------------------------- 行列生成 -----------------------------
    def matrix(self, progressbar, qApp):
        # --- 誘導速度係数 Q_ij をベネット&マイヤーズ式で評価 -------------
        N=len(self.y); Q=np.zeros((N,N))
        for i in range(N):
            if self.run: break
            for j in range(N):
                qApp.processEvents()
                progressbar.setValue(int((i*N+j+1)/(N*N)*100))
                yd =(self.y[i]-self.y[j])*np.cos(self.phi[j])+(self.z[i]-self.z[j])*np.sin(self.phi[j])
                zd =-(self.y[i]-self.y[j])*np.sin(self.phi[j])+(self.z[i]-self.z[j])*np.cos(self.phi[j])
                ydd=(self.y[i]+self.y[j])*np.cos(self.phi[j])-(self.z[i]-self.z[j])*np.sin(self.phi[j])
                zdd=(self.y[i]+self.y[j])*np.sin(self.phi[j])+(self.z[i]-self.z[j])*np.cos(self.phi[j])
                d  = self.dS[j]
                R2p,R2m = (yd-d)**2+zd**2 , (yd+d)**2+zd**2
                Rd2p,Rd2m= (ydd+d)**2+zdd**2 , (ydd-d)**2+zdd**2
                t1=((yd-d)/R2p-(yd+d)/R2m)*np.cos(self.phi[i]-self.phi[j])
                t2=(zd/R2p-zd/R2m)*np.sin(self.phi[i]-self.phi[j])
                t3=((ydd-d)/Rd2m-(ydd+d)/Rd2p)*np.cos(self.phi[i]+self.phi[j])
                t4=(zdd/Rd2m-zdd/Rd2p)*np.sin(self.phi[i]+self.phi[j])
                Q[i,j]=-1/(2*np.pi)*(t1+t2+t3+t4)
        self.Q_ij=Q

        # --- 多角形化行列 P --------------------------------------------
        self.polize_mat=np.zeros((len(self.y),self.n_section))
        self.polize_mat[:self.Ndiv_sec[1]+1,0]=1
        for j in range(1,self.n_section):
            a,b=self.y_section[j-1],self.y_section[j]
            idx= np.arange(self.Ndiv_sec[j-1]+1, self.Ndiv_sec[j]+1)
            self.polize_mat[idx,j-1]=-(np.array(self.y)[idx]-b)/(b-a)
            self.polize_mat[idx,j  ]= (np.array(self.y)[idx]-a)/(b-a)

        # 積分行列
        self.sh_mat=np.tril(np.ones((len(self.y),len(self.y))))*2
        np.fill_diagonal(self.sh_mat,1)
        self.sh_mat*= np.array(self.dS)[:,None]*self.U*self.rho
        self.mo_mat=self.sh_mat.copy()/ (self.U*self.rho)
        self.vd_mat=self.mo_mat/np.array(self.EI)[:,None]*1e6
        self.v_mat =self.mo_mat.copy()

        # 制約行列 B,C
        Bwant=np.zeros((1,len(self.y))); Bwant[0,-1]=1
        self.B = Bwant@self.v_mat@self.vd_mat@self.mo_mat@self.sh_mat@self.polize_mat
        self.B_val= self.max_tawami+ (Bwant@self.v_mat@self.vd_mat@self.mo_mat@self.sh_mat) @(
            np.array(self.sigma_wire).T/self.rho/self.U)
        self.C   = 4*self.rho*self.U*(np.array(self.dS)@self.polize_mat)
        self.C_val= self.M*9.8

    # ---------------------------- 最適化 (KKT) ---------------------------
    def optimize(self, checkbox):
        structural = checkbox.isChecked()
        A = self.Q_ij.copy()
        for j in range(A.shape[1]): A[:,j]*=np.array(self.dS)*2
        A=(A+A.T); A=self.rho*self.polize_mat.T@A@self.polize_mat

        if structural:
            A=np.vstack((A,-self.B,-self.C))
            A=np.column_stack((A, np.append(-self.B,[0,0])[:,None],
                                   np.append(-self.C,[0,0])[:,None]))
            rhs=np.zeros((A.shape[0],1)); rhs[-2,0]=-float(self.B_val); rhs[-1,0]=-float(self.C_val)
        else:
            A=np.vstack((A,-self.C))
            A=np.column_stack((A, np.append(-self.C,[0])[:,None]))
            rhs=np.zeros((A.shape[0],1)); rhs[-1,0]=-self.C_val

        sol=np.linalg.solve(A,rhs)
        self.gamma_opt = sol[:self.n_section]
        self.lambda1   = float(sol[self.n_section]) if structural else 0.0
        self.lambda2   = float(sol[self.n_section+(1 if structural else 0)])

        # 派生量
        self.bending_mat = self.v_mat@self.vd_mat@self.mo_mat@self.sh_mat
        delta = self.polize_mat@self.gamma_opt - np.array(self.sigma_wire)[:,None]/self.U/self.rho
        self.shearForce = self.sh_mat@delta
        self.moment     = self.mo_mat@self.shearForce
        self.bending_angle = self.vd_mat@self.moment
        self.bending    = self.bending_mat@delta
        rootg=self.M*9.8*4/np.pi/self.b/self.U/self.rho
        self.gamma_el=np.sqrt(np.maximum(rootg**2-(np.array(self.y)*rootg/self.b*2)**2,0))
        self.gamma   = self.polize_mat@self.gamma_opt
        self.ind_vel = (self.Q_ij/2)@self.gamma
        self.Di      = float(np.sum(self.rho*self.ind_vel*self.gamma*self.dy*2))
        self.Lift    = float(self.C@self.gamma_opt)
        self.comp=0


# ============================================================
#                          GUI main()
# ============================================================
def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("WM.ico"))

    splash = QSplashScreen(QPixmap("WM_splash.png")); splash.show()

    # ----- ウィジェット生成 ----------------------------------
    mainwindow      = QMainWindow(); mainwindow.setWindowTitle("Windmize")
    resulttabs      = ResultTabWidget()
    exeexportbutton = ExeExportButton()
    settingwidget   = SettingWidget()
    resultvalwidget = ResultValWidget()
    eisetdlg        = EIsettingWidget(settingwidget.tablewidget); eisetdlg.EIsetting(settingwidget.tablewidget)

    # ------- サンプル EI/σ ----------------------------------
    EI_sample=[3.4375e10,3.6671e10,1.6774e10,8.3058e9,1.8648e9,7.094e7]
    sig_sample=[0.377,0.357,0.284,0.245,0.0929,0.0440]
    for i in range(settingwidget.tablewidget.columnCount()-1):
        tbl=eisetdlg.EIinputWidget[i].EIinputtable
        for c in range(4):
            tbl.setItem(1,c+1,QTableWidgetItem(str(EI_sample[i])))
            tbl.setItem(2,c+1,QTableWidgetItem(str(sig_sample[i])))

    TR = TR797_modified()           # 数値エンジン

    # ----- コールバック ------------------------------------
    def insertcolumn():
        tw=settingwidget.tablewidget; col=tw.columnCount()
        tw.setColumnCount(col+1)
        tw.setHorizontalHeaderItem(col,QTableWidgetItem(f"第{col}翼"))
        tw.setItem(0,col,QTableWidgetItem(str(float(tw.item(0,col-1).text())+2000)))
        tw.setItem(1,col,QTableWidgetItem("1"))
        tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        eisetdlg.EIsetting(tw)              # タブ再生成

    def deletecolumn():
        tw=settingwidget.tablewidget; col=tw.columnCount()
        if col>=3:
            tw.setColumnCount(col-1)
            tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            eisetdlg.EIsetting(tw)

    def save_settings():
        cfg={
            "lift":settingwidget.lift_maxbending_input.liftinput.text(),
            "velocity":settingwidget.lift_maxbending_input.velocityinput.text(),
            "bending":settingwidget.lift_maxbending_input.bendinginput.text(),
            "wirepos":settingwidget.lift_maxbending_input.wireposinput.text(),
            "forcewire":settingwidget.lift_maxbending_input.forcewireinput.text(),
            "dy":settingwidget.lift_maxbending_input.dyinput.text(),
            "sections":[
                {"end":settingwidget.tablewidget.item(0,i+1).text(),
                 "coef":settingwidget.tablewidget.item(1,i+1).text()}
                for i in range(settingwidget.tablewidget.columnCount()-1)]
        }
        fname,_=QFileDialog.getSaveFileName(mainwindow,"設定を保存","","JSON (*.json)")
        if fname: json.dump(cfg,open(fname,"w",encoding="utf-8"),indent=2,ensure_ascii=False)

    def load_settings():
        fname, _ = QFileDialog.getOpenFileName(
            mainwindow, "設定を読み込む", "", "JSON (*.json)")
        if not fname:
            return
        try:
            with open(fname, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            QMessageBox.warning(mainwindow, "Error", f"JSON 読み込み失敗\n{e}")
            return

        lbl = settingwidget.lift_maxbending_input
        lbl.liftinput.setText(cfg.get("lift", ""))
        lbl.velocityinput.setText(cfg.get("velocity", ""))
        lbl.bendinginput.setText(cfg.get("bending", ""))
        lbl.wireposinput.setText(cfg.get("wirepos", ""))
        lbl.forcewireinput.setText(cfg.get("forcewire", ""))
        lbl.dyinput.setText(cfg.get("dy", ""))

        # --- セクション列数を合わせる --------------------------------------
        sec = cfg.get("sections", [])
        while len(sec) > settingwidget.tablewidget.columnCount() - 1:
            insertcolumn()
        for i, s in enumerate(sec):
            settingwidget.tablewidget.setItem(0, i + 1, QTableWidgetItem(s["end"]))
            settingwidget.tablewidget.setItem(1, i + 1, QTableWidgetItem(s["coef"]))

        # ========== 自動再計算 =============================================
        init_EI_widget()                            # EI タブを境界に追従
        exeexportbutton.progressbar.setValue(0)     # 進捗バーリセット

        TR.__init__()                               # エンジンをリセット
        TR.prepare(settingwidget, eisetdlg)         # 入力→内部データ
        TR.run = 0                                  # 計算フラグ = running
        TR.matrix(exeexportbutton.progressbar, app) # 行列生成
        TR.optimize(exeexportbutton.do_stracutual)  # KKT 解く
        show_results()                              # グラフ & 数値更新
        TR.run = 1                                  # idle 状態へ
        # ================================================================


    def init_EI_widget():
        yedge=[float(settingwidget.tablewidget.item(0,i+1).text())
               for i in range(settingwidget.tablewidget.columnCount()-1)]
        for i,gb in enumerate(eisetdlg.EIinputWidget):
            base=yedge[i] if i==0 else yedge[i]-yedge[i-1]
            for c,v in enumerate([base/4,base/2,base*3/4,base]):
                it=QTableWidgetItem(str(v)); it.setFlags(Qt.ItemIsSelectable|Qt.ItemIsEnabled)
                gb.EIinputtable.setItem(0,c+1,it)

    def show_results():
        # 循環
        resulttabs.circulation_graph.axes.clear()
        resulttabs.circulation_graph.drawplot(
            np.array(TR.y), TR.gamma, np.array(TR.y), TR.gamma_el,
            xlabel="y[m]",ylabel="gamma[m^2/s]",legend=("optimized","elliptical"),aspect="auto")
        # たわみ + ワイヤー線
        resulttabs.bending_graph.axes.clear()
        resulttabs.bending_graph.drawplot(np.array(TR.y),TR.bending,
                                          xlabel="y[m]",ylabel="bending[m]")
        if hasattr(TR,'Ndiv_wire') and TR.Ndiv_wire<len(TR.y):
            ywire=TR.y[TR.Ndiv_wire]; ax=resulttabs.bending_graph.axes
            ax.axvline(ywire,color='red',linestyle='--',label='Wire Pos.'); ax.legend(fontsize=10)
            resulttabs.bending_graph.draw()
        # 他のグラフ
        resulttabs.ind_graph.axes.clear()
        resulttabs.ind_graph.drawplot(np.array(TR.y), np.degrees(np.arctan(-TR.ind_vel/TR.U)),
                                      xlabel="y[m]",ylabel="induced angle[deg]",aspect="auto")
        resulttabs.bendingangle_graph.axes.clear()
        resulttabs.bendingangle_graph.drawplot(np.array(TR.y), np.degrees(TR.bending_angle),
                                               xlabel="y[m]",ylabel="bending angle[deg]")
        resulttabs.moment_graph.axes.clear()
        resulttabs.moment_graph.drawplot(np.array(TR.y),TR.moment,xlabel="y[m]",ylabel="moment[Nm]")
        resulttabs.shforce_graph.axes.clear()
        resulttabs.shforce_graph.drawplot(np.array(TR.y),TR.shearForce,xlabel="y[m]",ylabel="shearforce[N]")
        # 数値ラベル
        resultvalwidget.liftresultlabel.setText(f"計算揚力[kgf] : {TR.Lift/9.8:.3f}")
        resultvalwidget.Diresultlabel.setText(  f"   抗力[N] : {TR.Di:.3f}")
        resultvalwidget.swresultlabel.setText(  f"   桁重量概算[kg] : {TR.spar_weight:.3f}")
        resultvalwidget.lambda1label.setText(   f"   構造制約係数λ1[-] : "
                                                f"{TR.lambda1:.3f}" if exeexportbutton.do_stracutual.isChecked()
                                                else "   構造制約係数λ1[-] : --")
        resultvalwidget.lambda2label.setText(  f"   揚力制約係数λ2[-] : {TR.lambda2:.3f}")

    def calculation():
        if TR.run:
            exeexportbutton.exebutton.setText("計算中止")
            init_EI_widget(); TR.__init__(); TR.run=0
            TR.prepare(settingwidget,eisetdlg); TR.matrix(exeexportbutton.progressbar,app)
            if not TR.run: TR.optimize(exeexportbutton.do_stracutual); show_results()
            TR.run=1; exeexportbutton.exebutton.setText("計算")
        else:
            TR.run=1; exeexportbutton.exebutton.setText("計算")

    # ------------- 信号接続 --------------
    settingwidget.tablewidget.insertcolumn.clicked.connect(insertcolumn)
    settingwidget.tablewidget.deletecolumn.clicked.connect(deletecolumn)
    settingwidget.EIinput.EIinputbutton.clicked.connect(lambda: (init_EI_widget(), eisetdlg.show()))
    exeexportbutton.exebutton.clicked.connect(calculation)
    savebtn = QPushButton("設定保存"); loadbtn = QPushButton("設定読み込み")
    savebtn.clicked.connect(save_settings); loadbtn.clicked.connect(load_settings)

    # ------------- レイアウト ------------
    mainpanel=QWidget(); vlay=QVBoxLayout(mainpanel)
    vlay.addWidget(savebtn); vlay.addWidget(loadbtn)
    vlay.addWidget(resulttabs); vlay.addWidget(resultvalwidget)
    vlay.addWidget(exeexportbutton); vlay.addWidget(settingwidget)
    mainwindow.setCentralWidget(mainpanel)

    mainwindow.resize(1200,900); mainwindow.show(); splash.finish(mainwindow)
    sys.exit(app.exec_())


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
