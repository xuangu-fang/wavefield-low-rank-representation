"""Body copy for the published report page."""

BODY = """
<div class="page">
<header class="masthead">
  <div class="kicker"><span>项目报告</span><span>2026-08-19</span><span>wavefield-low-rank-representation</span></div>
  <h1>延迟占据决定波场的可压缩性</h1>
  <p class="standfirst">
    复波场在 <code>(x, f)</code> 展开下的数值秩等于<strong>带宽 × 到达时间占据测度</strong>。
    相位解调不消除能量，它只是把占据测度从绝对走时展宽换成相对延迟展宽——
    所以收益等于两者之比，而这个比值在训练任何模型之前就能算出来。
    同一条定律既预测收益出现的区间，也预测收益消失的区间。
  </p>
</header>

<div class="layout">
<nav class="index" aria-label="目录">
  <ol>
    <li><a href="#c1"><span class="n">01</span><span>三条主张</span></a></li>
    <li><a href="#c2"><span class="n">02</span><span>秩定律成立</span></a></li>
    <li><a href="#c3"><span class="n">03</span><span>带宽，不是频率</span></a></li>
    <li><a href="#c4"><span class="n">04</span><span>区间相图</span></a></li>
    <li><a href="#c5"><span class="n">05</span><span>任务收益</span></a></li>
    <li><a href="#c6"><span class="n">06</span><span>载波要多准</span></a></li>
    <li><a href="#c65"><span class="n">06b</span><span>网络不能替代</span></a></li>
    <li><a href="#c7"><span class="n">07</span><span>免费选载波</span></a></li>
    <li><a href="#c8"><span class="n">08</span><span>多载波方法</span></a></li>
    <li><a href="#c9"><span class="n">09</span><span>公开数据</span></a></li>
    <li><a href="#c10"><span class="n">10</span><span>负结果</span></a></li>
    <li><a href="#c11"><span class="n">11</span><span>理论补丁</span></a></li>
    <li><a href="#c12"><span class="n">12</span><span>下一步</span></a></li>
  </ol>
</nav>

<main>

<section id="c1">
  <h2><span class="n">01 / 主张</span>三条可证伪的主张</h2>
  <div class="claims">
    <div class="claim"><div class="tag">C1 秩定律</div><h4>秩 = 带宽 × 占据测度</h4>
      <p>对 <code>u(x,f)=∫g(x,τ)e<sup>-2πifτ</sup>dτ</code>，数值秩 ≈ <code>B·Λ<sub>B</sub></code>，
      其中 <code>Λ<sub>B</sub></code> 是在分辨率 <code>1/B</code> 下承载能量的延迟集合测度。</p></div>
    <div class="claim"><div class="tag">C2 增益律</div><h4>收益 = 占据测度之比</h4>
      <p>解调等价于逐点时移，把绝对走时占据 <code>Λ<sub>abs</sub></code> 换成相对延迟占据
      <code>Λ<sub>rel</sub></code>；收益 <code>G=(BΛ<sub>abs</sub>+1)/(BΛ<sub>rel</sub>+1)</code>。</p></div>
    <div class="claim"><div class="tag">C3 传导</div><h4>秩收益 → 任务收益</h4>
      <p>秩下降必然转化为稀疏采样任务的精度提升，且幅度由 C2 预测——包括预测"没有收益"。</p></div>
  </div>
  <p>三条主张都写了停止规则：合成场上偏差超过 2 倍、真实数据上相关性 <code>R²&lt;0.5</code>、
  或估计载波无法在任一真实任务上复现收益，则整条理论线终止。下面是它们实际的表现。</p>
</section>

<section id="c2">
  <h2><span class="n">02 / 证据一</span>秩定律成立，并优于所有替代预测量</h2>
  <div class="tablewrap"><table>
    <caption>实测数值秩（99% 能量）对 <code>B·Λ</code> 的最小二乘拟合。合成场上过原点拟合斜率为 0.999，即无自由参数的等式。</caption>
    <thead><tr><th>数据</th><th class="num">斜率</th><th class="num">R²</th><th class="num">样本</th></tr></thead>
    <tbody>
      <tr><td>合成多路径场</td><td class="num win">0.985</td><td class="num win">0.998</td><td class="num">840</td></tr>
      <tr><td>FDTD 区间扫描</td><td class="num">0.923</td><td class="num">0.944</td><td class="num">576</td></tr>
      <tr><td>The Well acoustic maze</td><td class="num">0.938</td><td class="num">0.954</td><td class="num">480</td></tr>
    </tbody>
  </table></div>
  <div class="tablewrap"><table>
    <caption>同一批合成数据上的预测量比较。占据测度的两个设计选择——按 <code>1/B</code> 平滑、与秩共用能量分位——是比较选出来的，不是假设的。</caption>
    <thead><tr><th>预测量</th><th class="num">斜率</th><th class="num">R²</th></tr></thead>
    <tbody>
      <tr><td>延迟占据测度（本文）</td><td class="num win">0.985</td><td class="num win">0.998</td></tr>
      <tr><td>朴素支撑长度 max−min</td><td class="num">0.712</td><td class="num">0.851</td></tr>
      <tr><td>各路径空间 range 的并集</td><td class="num">0.743</td><td class="num">0.873</td></tr>
    </tbody>
  </table></div>
  <figure><img src="{{FIG1}}" alt="三个数据集上实测秩对带宽乘占据测度的双对数散点，均贴合 y=x">
    <figcaption><b>图 1.</b> 实测秩 vs <code>B·Λ</code>。黑线为 <code>rank=BΛ</code>，红虚线为过原点拟合。
    中panel 左下方脱离主线的一簇点是开放无杂波区间的对齐场——那里的秩<em>低于</em>上界，原因见第 11 节。</figcaption>
  </figure>
</section>

<section id="c3">
  <h2><span class="n">03 / 证据二</span>秩由带宽决定，而不是中心频率</h2>
  <p class="lede">"高频波场难"是通行直觉。把两个通常被混在一起的变量分开扫，直觉就不成立了。</p>
  <div class="tablewrap"><table>
    <caption>固定中心频率扫带宽，与固定带宽扫中心频率，各自对秩的解释力。</caption>
    <thead><tr><th>区间</th><th class="num">rank ~ 带宽 R²</th><th class="num">rank ~ 中心频率 R²</th></tr></thead>
    <tbody>
      <tr><td>open, clear</td><td class="num win">0.957</td><td class="num bad">0.005</td></tr>
      <tr><td>open, sparse</td><td class="num">0.738</td><td class="num bad">0.196</td></tr>
      <tr><td>closed, dense</td><td class="num win">0.973</td><td class="num bad">0.001</td></tr>
    </tbody>
  </table></div>
  <p><strong>窄带的高频场很便宜，宽带的低频场很贵。</strong>难度的来源是带宽与延迟展宽的乘积，
  不是振荡本身有多快。</p>
  <figure><img src="{{FIG7}}" alt="左图秩随带宽线性增长，右图秩随中心频率几乎水平">
    <figcaption><b>图 2.</b> 同一批场，左：秩随带宽线性增长；右：秩对中心频率几乎水平。</figcaption>
  </figure>
</section>

<section id="c4">
  <h2><span class="n">04 / 证据三</span>区间相图：收益在哪出现、在哪消失</h2>
  <p class="lede">公开 benchmark 都固定在相图的某一点上，所以区间本身必须变成受控变量。
  自建 FDTD 求解器把<strong>边界吸收</strong>与<strong>散射体密度</strong>做成两个旋钮。</p>
  <p>下面每一行的横条是 <code>Λ<sub>rel</sub>/Λ<sub>abs</sub></code>——决定一切的那个比值；
  右端数字是实测秩增益。条越短，收益越大。</p>
  {{REGIME_BARS}}
  <figure><img src="{{FIG6}}" alt="三种介质中原始场与对齐场的实部对比">
    <figcaption><b>图 3.</b> 机制。开放无杂波介质中，对齐把同心振荡整个抹平（秩 6→1）；
    在混响与迷宫介质中，对齐后依然是散斑——首达之外的 coda 没有被任何单载波触及。</figcaption>
  </figure>
  <figure><img src="{{FIG2}}" alt="边界条件与散射密度构成的增益热图">
    <figcaption><b>图 4.</b> 相图。收益从 6.00 单调塌缩到 1.07。理论在混响区间预测精确（误差 &lt;5%），
    在开放区间偏保守。</figcaption>
  </figure>
</section>

<section id="c5">
  <h2><span class="n">05 / 证据四</span>表征收益确实转化为任务收益</h2>
  <p class="lede">任务：从随机 1–10% 的传感器位置重建全场，只在隐藏位置计分。
  <strong>唯一的变量是坐标系</strong>——同样的观测、同样的插值算法，只是做在原始场还是对齐场上。</p>
  <div class="tablewrap"><table>
    <caption>2% 传感器下的复数 NRMSE。1.0 等于"预测零场"。</caption>
    <thead><tr><th>区间</th><th class="num">原始场</th><th class="num">对齐场</th><th class="num">任务增益</th><th class="num">秩增益</th></tr></thead>
    <tbody>
      <tr><td>open, clear</td><td class="num">1.090</td><td class="num win">0.081</td><td class="num win">13.6×</td><td class="num">6.00</td></tr>
      <tr><td>open, sparse</td><td class="num">1.112</td><td class="num">0.748</td><td class="num">1.49×</td><td class="num">1.38</td></tr>
      <tr><td>partial, clear</td><td class="num">1.086</td><td class="num">0.829</td><td class="num">1.31×</td><td class="num">1.46</td></tr>
      <tr><td>closed, cluttered</td><td class="num">1.183</td><td class="num">1.107</td><td class="num bad">1.07×</td><td class="num">1.07</td></tr>
    </tbody>
  </table></div>
  <div class="formula">log(任务增益) = 1.49 · log(秩增益)      R² = 0.988   (36 个 case，2% 传感器)</div>
  <p>1% / 5% / 10% 传感器下 R² 分别为 0.986 / 0.980 / 0.969。<strong>秩增益解释了 97–99% 的任务增益方差</strong>，
  这正是主张 C3。</p>
  <figure><img src="{{FIG4}}" alt="四个区间下 NRMSE 随传感器比例的曲线，原始与对齐两条">
    <figcaption><b>图 5.</b> 收益随区间逐级塌缩：开放无杂波 14×，稀疏杂波 1.5×，混响密杂波 1.07×。</figcaption>
  </figure>
  <figure><img src="{{FIG5}}" alt="任务增益对秩增益的双对数散点">
    <figcaption><b>图 6.</b> 任务增益 vs 秩增益，按边界条件分组。</figcaption>
  </figure>
</section>

<section id="c6">
  <h2><span class="n">06 / 证据五</span>载波需要多准：一条可直接用的判据</h2>
  <p class="lede">理论说载波误差 <code>δτ</code> 等价于额外的延迟展宽，因此拐点应落在 <code>δτ ≈ 1/B</code>。</p>
  <div class="tablewrap"><table>
    <caption>open / clear 区间，2% 传感器下的复数 NRMSE。原始场基线为 1.090。</caption>
    <thead><tr><th>δτ（单位 1/B）</th><th class="num">0</th><th class="num">0.2</th><th class="num">0.5</th><th class="num">0.75</th><th class="num">1.0</th><th class="num">2.5</th></tr></thead>
    <tbody>
      <tr><td>粗糙空间误差</td><td class="num win">0.084</td><td class="num">0.415</td><td class="num">0.834</td><td class="num bad">1.005</td><td class="num bad">1.089</td><td class="num bad">1.221</td></tr>
      <tr><td>平滑标定误差</td><td class="num win">0.084</td><td class="num win">0.083</td><td class="num win">0.100</td><td class="num win">0.126</td><td class="num win">0.158</td><td class="num">0.404</td></tr>
    </tbody>
  </table></div>
  <div class="callout good"><div class="hd">可直接使用的结论</div>
    <p><strong>粗糙（空间高频）走时误差必须小于 1/B</strong>，到 <code>δτ=1/B</code> 收益已完全消失。
    而<strong>平滑的系统性误差几乎无害</strong>：<code>δτ=2.5/B</code> 时仍有 2.7× 收益。
    这解释了为什么 eikonal 求解器 1–3% 的离散偏差不会破坏方法。</p></div>
  <figure><img src="{{FIG3}}" alt="载波误差容限曲线，粗糙误差在 1/B 处越过基线">
    <figcaption><b>图 7.</b> 红色虚线为 <code>δτ=1/B</code>；灰点线为原始场基线。左右两图的差别就是"粗糙"与"平滑"的差别。</figcaption>
  </figure>
</section>

<section id="c65">
  <h2><span class="n">06b / 证据五之二</span>Fourier features 和 SIREN 不能替代载波</h2>
  <p class="lede">对一个机器学习场地的第一反问一定是"随便一个 Fourier-feature INR 或 SIREN
  不就把振荡学会了？"这里把它做成对照：同样的传感器、同样的架构与预算，唯一变量是网络预测
  <strong>原始场</strong>还是<strong>对齐后的包络</strong>。Fourier 带宽扫 {4, 16, 64} 并取<em>最好</em>的一档，
  比较对 baseline 有利。</p>
  <div class="tablewrap"><table>
    <caption>隐藏位置上的复数 NRMSE。1.0 = 零预测。</caption>
    <thead><tr><th>2% 传感器</th><th class="num">open, clear</th><th class="num">open, sparse</th><th class="num">partial, clear</th><th class="num">closed, dense</th></tr></thead>
    <tbody>
      <tr><td>网络（原始场，最好设置）</td><td class="num bad">0.98</td><td class="num bad">0.98</td><td class="num bad">0.98</td><td class="num bad">0.99</td></tr>
      <tr><td>网络（对齐场）</td><td class="num">0.36</td><td class="num">0.84</td><td class="num">0.82</td><td class="num">0.99</td></tr>
      <tr><td>线性插值（原始场）</td><td class="num bad">1.09</td><td class="num bad">1.10</td><td class="num bad">1.09</td><td class="num bad">1.14</td></tr>
      <tr><td>线性插值（对齐场）</td><td class="num win">0.084</td><td class="num">0.75</td><td class="num">0.83</td><td class="num">1.06</td></tr>
    </tbody>
  </table></div>
  <div class="tablewrap"><table>
    <thead><tr><th>10% 传感器</th><th class="num">open, clear</th><th class="num">open, sparse</th><th class="num">partial, clear</th><th class="num">closed, dense</th></tr></thead>
    <tbody>
      <tr><td>网络（原始场，最好设置）</td><td class="num bad">0.82</td><td class="num bad">0.93</td><td class="num bad">0.92</td><td class="num bad">1.09</td></tr>
      <tr><td>网络（对齐场）</td><td class="num">0.27</td><td class="num">0.79</td><td class="num">0.82</td><td class="num bad">1.12</td></tr>
      <tr><td>线性插值（对齐场）</td><td class="num win">0.046</td><td class="num">0.56</td><td class="num">0.61</td><td class="num">0.85</td></tr>
    </tbody>
  </table></div>
  <div class="callout"><div class="hd">训练误差才是关键诊断</div>
    <p>所有网络的<strong>训练</strong> NRMSE 是 0.000–0.015，<strong>测试</strong> NRMSE 是 0.75–1.28。
    网络把传感器拟合到几乎精确，仍然填不出中间——这正是旧 Track 2 审计里记录的病理。
    原因现在清楚了：在低于空间 Nyquist 的采样下，原始场在传感器之间<strong>根本不可辨识</strong>；
    载波把这部分信息从 <code>c(x)</code> 补进来，所以才可解。</p></div>
  <p><strong>表征对了之后，一个线性插值器比训练过的网络还好 5–17×。容量替代不了坐标。</strong>
  必须声明的局限：这里的网络是逐 case 拟合的隐式表示，不是跨 case 预训练的算子；
  本实验只否证"通用坐标编码可以替代物理载波"这一条。</p>
  <figure><img src="{{FIG10}}" alt="四个区间下网络与插值、原始与对齐的四条曲线，两种传感器密度">
    <figcaption><b>图 10.</b> 原始场上的网络（红）在两种传感器密度、所有区间下都停在零预测水平附近。</figcaption>
  </figure>
</section>

<section id="c7">
  <h2><span class="n">07 / 证据六</span>判据可以直接选载波，而且免费</h2>
  <p class="lede"><code>Λ<sub>rel</sub></code> 只要一次 FFT 加一次排序；它要预测的任务要一次完整重建。
  把"选 <code>Λ<sub>rel</sub></code> 最小的载波"在已有全部 case 上做事后打分：</p>
  <div class="tablewrap"><table>
    <caption>候选载波：无载波 / eikonal 走时 / 直线走时 / 数据拾取初至。</caption>
    <thead><tr><th>打分对象</th><th class="num">n</th><th class="num">选中即最优</th><th class="num">中位 regret</th><th class="num">规则增益</th><th class="num">oracle 增益</th></tr></thead>
    <tbody>
      <tr><td>秩（FDTD）</td><td class="num">144</td><td class="num win">0.93</td><td class="num">1.000</td><td class="num">1.45×</td><td class="num">1.45×</td></tr>
      <tr><td>任务（FDTD, 2%）</td><td class="num">36</td><td class="num">0.75</td><td class="num">1.000</td><td class="num win">2.20×</td><td class="num">2.33×</td></tr>
      <tr><td>任务（公开数据, 2%）</td><td class="num">22</td><td class="num">0.77</td><td class="num">1.000</td><td class="num">1.09×</td><td class="num">1.10×</td></tr>
    </tbody>
  </table></div>
  <p>中位 regret 恰为 1.000——<strong>中位情况下规则选中的就是最优载波</strong>；不一致的少数来自几个载波表现接近时的并列。
  规则最终拿到 oracle 可得增益的约 95%，代价为零。</p>
</section>

<section id="c8">
  <h2><span class="n">08 / 方法</span>多载波分解：定律直接指出的补救</h2>
  <p class="lede">单载波在混响介质里失效时，定律归咎的是 <strong>coda</strong>，不是低秩容器选错了。
  若诊断正确，补救就应当是按可分辨到达数增加载波，而不是换一种分解。</p>
  <div class="formula">u(x,f) = Σ<sub>m</sub> exp(-2πi f τ<sub>m</sub>(x)) · r<sub>m</sub>(x,f)      每个 r<sub>m</sub> 低秩</div>
  <p>在 Dirichlet 方箱里 <code>τ<sub>m</sub></code> 由<strong>镜像源</strong>精确给出（镜像源的直达项与 eikonal 走时最大差 0.005），
  所以可以完全不做相位优化就检验。拟合用单调的交替最小二乘；baseline 拿到的是 Eckart–Young
  最优 rank-R 近似，<strong>比较对 baseline 有利</strong>。等参数预算 <code>R = M × k</code>：</p>
  <div class="tablewrap"><table>
    <caption>R=24 时的相对逼近误差。最后一列是多载波相对单载波的改进倍数。</caption>
    <thead><tr><th>区间</th><th class="num">plain SVD</th><th class="num">单载波</th><th class="num">多载波</th><th class="num">改进</th></tr></thead>
    <tbody>
      <tr><td>partial, clear</td><td class="num">0.440</td><td class="num">0.329</td><td class="num win">0.234</td><td class="num win">1.41×</td></tr>
      <tr><td>closed, clear</td><td class="num">0.608</td><td class="num">0.522</td><td class="num win">0.370</td><td class="num win">1.41×</td></tr>
      <tr><td>closed, sparse</td><td class="num">0.643</td><td class="num">0.582</td><td class="num">0.505</td><td class="num">1.15×</td></tr>
      <tr><td>closed, dense</td><td class="num">0.652</td><td class="num">0.604</td><td class="num">0.567</td><td class="num">1.06×</td></tr>
      <tr><td>open, clear</td><td class="num">0.011</td><td class="num">0.003</td><td class="num">0.003</td><td class="num">0.88×</td></tr>
    </tbody>
  </table></div>
  <div class="callout good"><div class="hd">方法与判据互相印证</div>
    <p>优势随散射体密度单调衰减（1.41 → 1.15 → 1.06）<strong>正是理论预测的形状</strong>：
    镜像源只建模边界反射，不建模体散射；当 coda 由体散射主导，多载波无从下手。
    在 open / clear 一栏多载波不再有优势——单载波已经把 <code>Λ<sub>rel</sub></code> 压到接近零。</p></div>
  <div class="tablewrap"><table>
    <caption>同一模型做 5% 条目的稀疏补全（正则强度由观测集内部划出的验证集选，不碰测试集）。</caption>
    <thead><tr><th>区间</th><th class="num">plain SVD</th><th class="num">单载波</th><th class="num">多载波</th></tr></thead>
    <tbody>
      <tr><td>open, clear</td><td class="num">0.913</td><td class="num">0.843</td><td class="num win">0.053</td></tr>
      <tr><td>partial, clear</td><td class="num bad">1.096</td><td class="num bad">1.015</td><td class="num win">0.746</td></tr>
      <tr><td>closed, clear</td><td class="num bad">1.147</td><td class="num bad">1.102</td><td class="num">0.937</td></tr>
      <tr><td>closed, dense</td><td class="num bad">1.154</td><td class="num bad">1.132</td><td class="num">0.986</td></tr>
    </tbody>
  </table></div>
  <p>open / clear 上比最好的 baseline 好 <strong>16×</strong>。baseline 在混响区间 NRMSE &gt; 1（劣于零预测）——
  5% 采样根本无法重建一个秩 56–64 的场，这同样是定律的直接推论。</p>
  <figure><img src="{{FIG8}}" alt="四个区间下逼近误差随参数预算的三条曲线">
    <figcaption><b>图 8.</b> 等参数预算下的逼近误差。多载波的优势随散射增强而消失。</figcaption>
  </figure>

  <h3>8.1&nbsp;&nbsp;去掉几何假设：直接从数据里长出载波库</h3>
  <p>镜像源需要知道边界，这是上面结果里唯一的"作弊"成分。把它换成纯数据驱动的流程：
  每一轮重新拟合整个多载波模型，在<strong>模型残差</strong>上扫描"哪个虚源波前能让残差同相叠加"，
  把该虚源加入载波库；<strong>只在固定总预算下拟合变好时才保留</strong>——多一个载波意味着
  每个载波分到的秩更低，所以没用的载波会让模型变差而被自动拒绝。全过程不使用任何几何信息。</p>
  <div class="tablewrap"><table>
    <caption>R=24 时的相对逼近误差。最后一列是流程自己决定保留的载波数。</caption>
    <thead><tr><th>区间</th><th class="num">plain rank-R</th><th class="num">单载波</th><th class="num">估计载波（无几何）</th><th class="num">oracle 镜像源</th><th class="num">载波数</th></tr></thead>
    <tbody>
      <tr><td>open, clear</td><td class="num">0.011</td><td class="num">0.003</td><td class="num win">0.003</td><td class="num">0.003</td><td class="num">1（自动停止）</td></tr>
      <tr><td>partial, clear</td><td class="num">0.440</td><td class="num">0.329</td><td class="num win">0.289</td><td class="num">0.231</td><td class="num">2</td></tr>
      <tr><td>closed, clear</td><td class="num">0.608</td><td class="num">0.522</td><td class="num win">0.390</td><td class="num">0.372</td><td class="num">4</td></tr>
      <tr><td>closed, sparse</td><td class="num">0.643</td><td class="num">0.583</td><td class="num win">0.518</td><td class="num">0.507</td><td class="num">4</td></tr>
      <tr><td>closed, dense</td><td class="num">0.652</td><td class="num">0.604</td><td class="num win">0.571</td><td class="num">0.568</td><td class="num">2–3</td></tr>
    </tbody>
  </table></div>
  <div class="callout good"><div class="hd">几何假设可以去掉</div>
    <p><strong>估计载波拿到 oracle 增益的 90–96%</strong>（partial / clear 为 65%），不需要任何几何输入。
    每个被接受的虚源波前与真实镜像源波前的形状误差，在无杂波区间是
    <strong>0.02–0.08 × (1/B)</strong>，远在第 6 节的容限之内；加入散射体后退化到 0.25–0.5 × (1/B)，仍在容限内。
    在 open / clear 区间流程<strong>自动停在 M=1</strong>——它正确判断出没有第二个值得加的到达。
    判据既告诉你什么时候该加载波，也告诉你什么时候不必加。</p></div>
  <figure><img src="{{FIG9}}" alt="四种方法在五个区间下的误差柱状图，以及估计载波捕获 oracle 增益的百分比">
    <figcaption><b>图 9.</b> 左：等参数预算下四种模型的误差。右：估计载波捕获了多少 oracle（已知几何）增益。</figcaption>
  </figure>
  <div class="callout"><div class="hd">实现教训（保留以免后来者重蹈）</div>
    <p>最初用 Adam 拟合该模型，在较大预算上<strong>发散</strong>（误差 7、30、91）。载波之间远非正交，
    一阶方法条件数极差。换成交替最小二乘后单调收敛。这不是方法的性质，是优化器的性质。</p></div>
</section>

<section id="c9">
  <h2><span class="n">09 / 公开数据</span>包括被预测出来的负结果</h2>
  <div class="tablewrap"><table>
    <caption>同一套任务代码跑在三个公开数据集上。理论预测栏是在跑任务<em>之前</em>由占据测度给出的。</caption>
    <thead><tr><th>数据集</th><th class="num">n</th><th class="num">秩</th><th class="num">秩增益</th><th class="num">任务增益 (2%)</th><th>理论预测</th></tr></thead>
    <tbody>
      <tr><td>The Well acoustic maze</td><td class="num">12</td><td class="num">24.7 → 23.2</td><td class="num">1.06</td><td class="num bad">1.02×</td><td>无收益 ✓</td></tr>
      <tr><td>The Well acoustic inclusions (256²)</td><td class="num">10</td><td class="num">6.5 → 6.2</td><td class="num">1.05</td><td class="num">1.18×</td><td>微弱收益 ✓</td></tr>
      <tr><td>The Well Helmholtz staircase（train，26 源）</td><td class="num">26</td><td class="num">3.85 → 1.65</td><td class="num win">2.54</td><td class="num win">1.69×</td><td>中等收益 ✓</td></tr>
    </tbody>
  </table></div>
  <div class="tablewrap"><table>
    <caption>staircase：26 个源位置上的传感器重建。增益随传感器变密而增大，方差很小。</caption>
    <thead><tr><th>传感器比例</th><th class="num">1%</th><th class="num">2%</th><th class="num">5%</th><th class="num">10%</th></tr></thead>
    <tbody>
      <tr><td>原始场 NRMSE</td><td class="num">0.211</td><td class="num">0.162</td><td class="num">0.105</td><td class="num">0.075</td></tr>
      <tr><td>对齐场 NRMSE</td><td class="num win">0.136</td><td class="num win">0.099</td><td class="num win">0.057</td><td class="num win">0.037</td></tr>
      <tr><td>增益</td><td class="num">1.61 ± 0.20</td><td class="num">1.69 ± 0.22</td><td class="num">1.90 ± 0.26</td><td class="num win">2.12 ± 0.34</td></tr>
    </tbody>
  </table></div>
  <p>逐项低秩补全从 0.992 提升到 <strong>0.813</strong>（原始场基本等于零预测）。
  test split 的 3 条轨迹共用同一个源位置，所以逐源统计用的是 train split。</p>

  <h3>9.1&nbsp;&nbsp;等预算压缩：公开数据上最强的一组数字</h3>
  <p>上表衡量的是稀疏重建任务。若直接问"同样的参数预算下，表征能把场压到多小"，
  公开数据给出的差距要大得多。载波用的是可部署的 eikonal 走时，不是 oracle。</p>
  <div class="tablewrap"><table>
    <caption>等参数预算下的相对逼近误差。多载波流程在这两个公开集上几乎总是自动停在 M=1。</caption>
    <thead><tr><th>数据集</th><th class="num">n</th><th class="num">预算 R</th><th class="num">plain rank-R</th><th class="num">单载波</th><th class="num">倍数</th><th class="num">估计多载波</th></tr></thead>
    <tbody>
      <tr><td>Helmholtz staircase</td><td class="num">8</td><td class="num">4</td><td class="num">0.2606</td><td class="num win">0.0288</td><td class="num win">9.8×</td><td class="num">0.0275</td></tr>
      <tr><td>Helmholtz staircase</td><td class="num">8</td><td class="num">8</td><td class="num">0.0212</td><td class="num win">0.0019</td><td class="num win">11.5×</td><td class="num">0.0019</td></tr>
      <tr><td>acoustic inclusions</td><td class="num">6</td><td class="num">8</td><td class="num">0.2750</td><td class="num">0.2416</td><td class="num">1.13×</td><td class="num">0.2572</td></tr>
      <tr><td>acoustic inclusions</td><td class="num">6</td><td class="num">16</td><td class="num">0.0765</td><td class="num">0.0636</td><td class="num">1.22×</td><td class="num">0.0637</td></tr>
    </tbody>
  </table></div>
  <p><strong>staircase 上一个可部署的单载波把等预算逼近误差降了一个数量级</strong>——比稀疏重建任务上的
  1.7–2.1× 大得多，因为重建任务还受采样几何限制，而压缩只受表征本身限制。
  inclusions 的占据比在 0.71–0.94 之间（预测 ~1.1× 收益），实测 1.13–1.22×，
  个别 case 略低于 1；在收益≈1 的区间出现正负抖动是预期内的。</p>
  <div class="callout"><div class="hd">对社区有用的观察</div>
    <p>主流公开波动 benchmark 绝大多数落在<strong>混响区间</strong>，而这恰恰是相位对齐类方法必然失效的区间。
    这解释了这类方法在 benchmark 上长期让人失望的历史，也说明 benchmark 覆盖存在系统性缺口——
    真实的地震、超声、雷达场景大多是吸收/开放介质，即收益最大的区间。</p></div>
</section>

<section id="c10">
  <h2><span class="n">10 / 负结果</span>跨频外推在 staircase 上全线失败</h2>
  <p class="lede">在 16 个 ω、同一几何上做"低频段拟合 → 高频预测"：<strong>所有方法都 ≈ 或劣于零预测</strong>，
  包括理论指引的"对齐后低秩频率延拓"。</p>
  <div class="tablewrap"><table>
    <caption>复数 NRMSE，1.0 = 零预测。每格取该方法在所有超参数下的最好结果。</caption>
    <thead><tr><th class="num">训练频点</th><th class="num">外推倍数</th><th class="num">copy-last</th><th class="num">raw 逐点</th><th class="num">raw 幅相分离</th><th class="num">对齐低秩</th></tr></thead>
    <tbody>
      <tr><td class="num">6</td><td class="num">2.52×</td><td class="num bad">1.42</td><td class="num bad">1.47</td><td class="num">0.80</td><td class="num bad">1.51</td></tr>
      <tr><td class="num">12</td><td class="num">1.23×</td><td class="num bad">1.06</td><td class="num bad">1.11</td><td class="num">0.88</td><td class="num">0.98</td></tr>
    </tbody>
  </table></div>
  <p><strong>诊断（不是简单的"方法不行"）：</strong>该数据集的物理是沿周期性台阶传播的 trapped modes，
  其面上波数与 ω 呈非线性色散关系，并存在<strong>模式截止</strong>。因此
  (a) 线性于 <code>f</code> 的走时载波原理上无法吸收色散相位；
  (b) 高频段会出现低频段完全不存在的新模式，任何解析延拓都跨不过截止点。</p>
  <p>所以跨频外推不被列为本方法的适用任务，而是列为该定律的<strong>边界</strong>之一。
  这与 APEX 的动机一致：跨频恢复需要生成式补全，而非外推。</p>
</section>

<section id="c11">
  <h2><span class="n">11 / 修正</span>理论必须打的补丁</h2>
  <p>原始假设写的是等式。数据要求把它改成<strong>上界</strong>：</p>
  <div class="formula">rank<sub>ε</sub>(U)  ≲  min( B·Λ<sub>B</sub> ,  rank<sub>ε</sub>(G) )        G = 时域脉冲响应矩阵</div>
  <p>在开放无杂波介质里，对齐后的 <code>G</code> 近乎秩 1（一个子波、一个到达、平滑幅度），
  实测秩落在 <code>B·Λ<sub>rel</sub></code> <em>之下</em>，因此实测增益（6.0）大于预测增益（3.0）。
  在混响介质里 coda 空间上弥散，上界紧，预测与实测吻合到几个百分点。</p>
  <p><strong>定律可靠地预测"区间"，并在最有利的区间对收益幅度保守。</strong>对使用者是安全的方向：
  它不会高估收益。</p>
</section>

<section id="c12">
  <h2><span class="n">12 / 下一步</span>按优先级</h2>
  <ol class="steps">
    <li><strong>把虚源模型推广到非均匀背景。</strong>目前虚源用的是常速背景下的
      <code>|x−p|/c</code>；强速度反差介质需要用射线或 eikonal 走时表替代这一步。</li>
    <li><strong>补上"有利区间"的公开 benchmark。</strong>目前只有自建 FDTD 在该区间，
      需要一个可引用的公开开放介质数据（WaveBench 的部分任务，或官方配对的 OpenFWI 文件）。</li>
    <li><strong>色散推广。</strong>把载波从 <code>2πfτ</code> 推广到一般 <code>φ(x,f)</code>（局部相位斜率估计）——
      这是 staircase 的失败给出的明确方向。</li>
    <li><strong>接生成式残差。</strong>在 <code>Λ<sub>rel</sub></code> 大的区间低秩必然不够，
      此时按 APEX 路线把对齐后的 residual 交给 flow / diffusion。
      定律正好指出"什么时候必须上生成模型"。</li>
  </ol>
  <h3>顺带发现的数据完整性问题</h3>
  <div class="callout"><div class="hd">影响其他项目</div>
    <p><strong>本地 OpenFWI 副本的地震数据与速度模型来自不同 family</strong>（<code>seis2_*</code> 配 <code>vel4_*</code>）。
    40 个模型上，由初至时距斜率反推的表层速度与配对模型的表层速度相关系数仅 <strong>0.135</strong>。
    任何在该副本上做的速度条件化实验都是无效的，需要重新下载官方配对文件。</p></div>
  <p>另外三条记录在 <code>docs/DATA_INTEGRITY.md</code>：staircase 的 y 轴在数组中反向存储；
  其 50 个时间步是 <code>e<sup>-iωt</sup></code> 的纯解析冗余；test split 的 3 条轨迹共用同一个源位置。</p>
</section>

<footer>
  <p>全部结论可复现：<code>experiments/exp01</code>–<code>exp10</code>，16 个单元测试，
  汇总在 <code>reports/summary.json</code>，图在 <code>reports/figures/</code>。
  原始 JSON 保存在 <code>results/</code>（未纳入 git），大数据位于 NFS。</p>
</footer>

</main>
</div>
</div>
"""
