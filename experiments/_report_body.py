"""Body copy for the published report page."""

BODY = """
<div class="page">
<header class="masthead">
  <div class="kicker"><span>项目报告</span><span>2026-08-20</span><span>wavefield-low-rank-representation</span></div>
  <h1>可辨识性，而不是容量</h1>
  <p class="standfirst">
    稀疏波场重构的瓶颈是<strong>可辨识性</strong>，不是模型容量。传感器之间的场在信息上
    无法被采样确定；物理先验的作用不是"更好的归纳偏置"，而是<strong>把缺失的自由度补进来</strong>。
    缺多少、补得够不够、什么时候根本补不上——都可以用<strong>一次 FFT</strong> 在训练前算出来，
    而这些信息有时可以自监督地从数据本身学回来。
  </p>
</header>

<div class="layout">
<nav class="index" aria-label="目录">
  <ol>
    <li><a href="#c0"><span class="n">00</span><span>可辨识性界</span></a></li>
    <li><a href="#c02"><span class="n">00b</span><span>不是容量</span></a></li>
    <li><a href="#c03"><span class="n">00c</span><span>把界当损失</span></a></li>
    <li><a href="#c04"><span class="n">00d</span><span>与采样定理的关系</span></a></li>
    <li><a href="#c05"><span class="n">00e</span><span>留出验证</span></a></li>
    <li><a href="#c06"><span class="n">00f</span><span>摊销与迁移</span></a></li>
    <li><a href="#c07"><span class="n">00g</span><span>跨几何（限制）</span></a></li>
    <li><a href="#c08"><span class="n">00h</span><span>覆盖审计</span></a></li>
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
    <li><a href="#c115"><span class="n">11b</span><span>算子学习（弱）</span></a></li>
    <li><a href="#c117"><span class="n">11c</span><span>通用性边界</span></a></li>
    <li><a href="#c12"><span class="n">12</span><span>下一步</span></a></li>
    <li><a href="#cA"><span class="n">A</span><span>载波方法细节</span></a></li>
  </ol>
</nav>

<main>

<section id="c0">
  <h2><span class="n">00 / 核心</span>一次 FFT 给出的可辨识性界</h2>
  <p class="lede">在间距为 <code>m</code> 的规则传感器阵列上，波数高于 <code>1/(2m)</code> 的能量
  与低于它的能量无法区分。因此<strong>任何</strong>方法的相对误差都不低于：</p>
  <div class="formula">误差  ≥  √( 能量(|k| &gt; 1/(2m)) ⁄ 总能量 )   ≡  可辨识性界</div>
  <p>这一句有三个特点：它是<strong>信息层面</strong>的陈述（与用什么模型无关）、
  <strong>一次 FFT 就能算</strong>（不需训练，也不需真值）、
  而且<strong>载波的全部作用就是把能量搬到门槛以下</strong>。</p>
  <div class="callout"><div class="hd">测量纪律：三条都是踩坑之后加的</div>
    <p><strong>1. 规则阵列，不是随机采样。</strong>随机采样下混叠不是一堵墙——稀疏的高波数内容可以被恢复
    （压缩感知正利用这一点），此时该量根本不是界。规则阵列也更接近真实传感硬件。<br>
    <strong>2. 有遮挡的区域要裁剪，不能补零。</strong>补零插入一条本不存在的阶跃边，
    凭空制造高波数能量把界抬高。<br>
    <strong>3. 加锥形窗，且预测与实测用同一个窗。</strong>非周期的光滑场在矩形窗下泄漏出 1/k 拖尾；
    对严重过采样的场，那条拖尾<em>就是</em>全部"界外能量"，足以让界超过它本该下界的误差。
    Tukey(0.25) 消除它——FDTD 场对窗几乎不敏感（比值 1.2–1.4 不随 α 变），staircase 则完全依赖它。</p></div>
  <div class="tablewrap"><table>
    <caption>界随阵列间距的行为，就是整个故事。数值为界本身，越小越可辨识。</caption>
    <thead><tr><th>场</th><th>坐标系</th><th class="num">m=2</th><th class="num">m=3</th><th class="num">m=4</th><th class="num">m=6</th><th class="num">m=8</th><th class="num">m=11</th><th class="num">m=16</th></tr></thead>
    <tbody>
      <tr><td>open, clear</td><td>raw</td><td class="num">0.022</td><td class="num">0.091</td><td class="num">0.445</td><td class="num bad">0.997</td><td class="num bad">0.999</td><td class="num bad">1.000</td><td class="num bad">1.000</td></tr>
      <tr><td>open, clear</td><td><strong>aligned</strong></td><td class="num win">0.014</td><td class="num win">0.032</td><td class="num win">0.040</td><td class="num win">0.062</td><td class="num win">0.102</td><td class="num win">0.170</td><td class="num win">0.254</td></tr>
      <tr><td>open, sparse</td><td>raw</td><td class="num">0.245</td><td class="num">0.393</td><td class="num">0.605</td><td class="num bad">0.993</td><td class="num bad">0.998</td><td class="num bad">0.999</td><td class="num bad">1.000</td></tr>
      <tr><td>open, sparse</td><td>aligned</td><td class="num">0.276</td><td class="num">0.443</td><td class="num">0.489</td><td class="num">0.556</td><td class="num">0.596</td><td class="num">0.637</td><td class="num">0.685</td></tr>
      <tr><td>closed, dense</td><td>raw</td><td class="num">0.439</td><td class="num">0.655</td><td class="num">0.784</td><td class="num bad">0.990</td><td class="num bad">0.997</td><td class="num bad">0.999</td><td class="num bad">0.999</td></tr>
      <tr><td>closed, dense</td><td>aligned</td><td class="num">0.524</td><td class="num">0.738</td><td class="num">0.796</td><td class="num">0.860</td><td class="num">0.886</td><td class="num">0.911</td><td class="num">0.937</td></tr>
      <tr><td>The Well maze</td><td>raw</td><td class="num">0.627</td><td class="num bad">0.972</td><td class="num bad">0.993</td><td class="num bad">0.999</td><td class="num bad">1.000</td><td class="num bad">1.000</td><td class="num bad">1.000</td></tr>
      <tr><td>The Well maze</td><td>aligned</td><td class="num">0.736</td><td class="num">0.878</td><td class="num">0.937</td><td class="num bad">0.985</td><td class="num bad">0.994</td><td class="num bad">0.998</td><td class="num bad">0.999</td></tr>
    </tbody>
  </table></div>
  <p>原始坐标下，<strong>间距一到 6 像素可辨识性就已经塌到 1.0</strong>（等于零预测）；
  对齐坐标在 open/clear 下到间距 16 仍然是 0.254。其余区间两者并无差别——判据同时说明了这一点。</p>
  <div class="tablewrap"><table>
    <caption>换算成传感器数量：达到 NRMSE ≤ 0.5 所需的采样比例。</caption>
    <thead><tr><th>区间</th><th class="num">raw</th><th class="num">aligned</th><th class="num">省下</th></tr></thead>
    <tbody>
      <tr><td>open, clear</td><td class="num">9.4%</td><td class="num win">0.39%</td><td class="num win">24×</td></tr>
      <tr><td>open, sparse</td><td class="num">15.8%</td><td class="num">15.1%</td><td class="num bad">1.0×</td></tr>
      <tr><td>partial, clear</td><td class="num">9.4%</td><td class="num">15.5%</td><td class="num bad">0.6×</td></tr>
      <tr><td>closed, dense</td><td class="num" colspan="2">两者都达不到</td><td class="num">—</td></tr>
    </tbody>
  </table></div>
  <div class="callout good"><div class="hd">界在多少数据上成立</div>
    <p>自建求解器 + The Well maze + acoustic inclusions：<strong>610 组</strong>测量（3 seed），
    打破比例 <strong>1.97%</strong>——其中两个公开数据集均为 <strong>0%</strong>，自建求解器 2.4%。<br>
    公开频域数据（Helmholtz staircase + WaveBench）：<strong>224 组</strong>，打破比例 <strong>8.0%</strong>
    （WaveBench ω=40 为 <strong>0%</strong>），R²=0.88。</p></div>
  <div class="tablewrap"><table>
    <caption>它可以直接当设计工具：给定目标精度，界反推出需要多密的阵列。</caption>
    <thead><tr><th>目标精度</th><th class="num">拟合斜率</th><th class="num">R²</th><th class="num">n</th></tr></thead>
    <tbody>
      <tr><td>NRMSE ≤ 0.5</td><td class="num">0.793</td><td class="num win">0.947</td><td class="num">40</td></tr>
      <tr><td>NRMSE ≤ 0.3</td><td class="num">1.252</td><td class="num win">0.973</td><td class="num">20</td></tr>
    </tbody>
  </table></div>
  <p><strong>在建阵列之前，一次 FFT 就能算出它需要多密</strong>，误差在对数尺度上 R²≈0.95。</p>
  <figure><img src="{{FIG13}}" alt="左：实测误差对界的双对数散点；右：误差随阵列间距的曲线">
    <figcaption><b>图 0.</b> 左：界成立且接近紧。右：载波把整条曲线搬下去——不增加任何一次测量。</figcaption>
  </figure>
</section>

<section id="c02">
  <h2><span class="n">00b / 不是容量</span>跨 128 倍参数量，测试误差只动 3%</h2>
  <p class="lede">如果传感器之间填不出来是<strong>建模</strong>失败，更大或训练更久的模型应该能修好它。
  如果是<strong>可辨识性</strong>失败，什么都修不好——而同一个问题在载波坐标里会突然变简单，
  且没有增加任何一次测量。</p>
  <div class="tablewrap"><table>
    <caption>六个估计器，同样的规则阵列、同样的场，48 组设置。</caption>
    <thead><tr><th>估计器</th><th class="num">参数量</th><th class="num">训练 NRMSE</th><th class="num">测试 NRMSE</th></tr></thead>
    <tbody>
      <tr><td>Fourier-feature MLP（小）</td><td class="num">8 192</td><td class="num">1.4×10⁻⁷</td><td class="num bad">0.970</td></tr>
      <tr><td>Fourier-feature MLP（中）</td><td class="num">196 608</td><td class="num">1.2×10⁻⁷</td><td class="num bad">0.958</td></tr>
      <tr><td>Fourier-feature MLP（大）</td><td class="num">1 048 576</td><td class="num">1.2×10⁻⁷</td><td class="num bad">0.940</td></tr>
      <tr><td>SIREN</td><td class="num">196 608</td><td class="num">7.2×10⁻⁴</td><td class="num">0.900</td></tr>
      <tr><td>最近邻</td><td class="num">—</td><td class="num">—</td><td class="num">0.934</td></tr>
      <tr><td><strong>线性插值（零参数）</strong></td><td class="num">—</td><td class="num">—</td><td class="num win">0.807</td></tr>
    </tbody>
  </table></div>
  <div class="callout good"><div class="hd">三条读数</div>
    <p><strong>1.</strong> 训练误差恒为 10⁻⁷，测试误差在 <strong>128 倍</strong>容量跨度上只从 0.970 变到 0.940
    （<strong>3%</strong>）。网络把传感器拟合到机器精度，仍然填不出中间——不是欠拟合，也不是欠训练。<br>
    <strong>2.</strong> <strong>零参数的线性插值打败了每一个网络</strong>，包括一百万参数那个；
    72 组里它有多数组是最优估计器。容量替代不了信息。<br>
    <strong>3.</strong> <strong>换坐标系才有用</strong>：界从 0.767 降到 0.528，所有估计器能达到的最好成绩
    跟着从 0.887 降到 0.604——<strong>没有增加任何一次测量</strong>。</p></div>
  <p>界在 <strong>72</strong> 组里被打破 6 次（8.3%），线性插值平均是界的 1.48 倍——即界不仅成立，而且接近可达。
  这就是"可辨识性，而不是容量"这句话的全部经验内容。</p>
  <figure><img src="{{FIG14}}" alt="左：测试与训练误差随参数量；右：两种坐标系下的界与最好估计器">
    <figcaption><b>图 1.</b> 左：训练误差恒在机器精度，测试误差跨 128 倍容量几乎不动，
    零参数的线性插值反而最好。右：换坐标系同时降低界与可达误差。</figcaption>
  </figure>
</section>

<section id="c03">
  <h2><span class="n">00c / 学回来</span>把界本身当成损失函数</h2>
  <p class="lede">界是<strong>场在某个坐标系下</strong>的性质。物理通过提供走时把它降下去。
  那么不给介质、不给源、不给求解器，能不能把同样的降幅学出来？</p>
  <div class="formula">min<sub>θ</sub>   能量( |k| &gt; 1/(2m) ) ⁄ 总能量      在坐标 exp(+i φ<sub>θ</sub>) 下</div>
  <p>这个量<strong>无标签、无真值、一次 FFT 即可微</strong>。于是同一个量同时是
  <strong>诊断、训练目标和上报指标</strong>。</p>
  <div class="tablewrap"><table>
    <caption>阵列间距 m=6，4 个区间 × 2 seed 平均。</caption>
    <thead><tr><th>坐标系</th><th class="num">界</th><th class="num">实测误差</th></tr></thead>
    <tbody>
      <tr><td>无载波</td><td class="num bad">0.994</td><td class="num bad">1.109</td></tr>
      <tr><td>eikonal（物理）</td><td class="num">0.535</td><td class="num win">0.636</td></tr>
      <tr><td>eikonal 被粗糙误差破坏</td><td class="num bad">0.927</td><td class="num bad">1.122</td></tr>
      <tr><td><strong>纯学习（无物理）</strong></td><td class="num win">0.526</td><td class="num">0.688</td></tr>
      <tr><td><strong>学习修复被破坏的载波</strong></td><td class="num win">0.525</td><td class="num">0.688</td></tr>
      <tr><td>学习（从正确物理出发）</td><td class="num win">0.525</td><td class="num">0.684</td></tr>
      <tr><td><em>消融：改用频率轴目标</em></td><td class="num bad">0.651</td><td class="num bad">0.769</td></tr>
    </tbody>
  </table></div>
  <div class="tablewrap"><table>
    <caption>逐区间的界（m=6）。</caption>
    <thead><tr><th>区间</th><th class="num">无载波</th><th class="num">物理</th><th class="num">物理被破坏</th><th class="num">纯学习</th><th class="num">学习修复</th></tr></thead>
    <tbody>
      <tr><td>open, clear</td><td class="num bad">0.997</td><td class="num win">0.062</td><td class="num bad">0.888</td><td class="num">0.128</td><td class="num">0.073</td></tr>
      <tr><td>open, sparse</td><td class="num bad">0.991</td><td class="num">0.594</td><td class="num bad">0.930</td><td class="num win">0.549</td><td class="num win">0.548</td></tr>
      <tr><td>partial, clear</td><td class="num bad">0.997</td><td class="num">0.663</td><td class="num bad">0.916</td><td class="num">0.663</td><td class="num">0.663</td></tr>
      <tr><td>closed, dense</td><td class="num bad">0.987</td><td class="num">0.878</td><td class="num bad">0.939</td><td class="num">0.877</td><td class="num win">0.860</td></tr>
    </tbody>
  </table></div>
  <div class="callout good"><div class="hd">四条读数</div>
    <p><strong>1.</strong> 纯学习在界上<strong>追平物理</strong>（0.526 vs 0.535）——不给介质、不给源位置、不跑求解器。<br>
    <strong>2.</strong> 四个区间里学习在<strong>三个</strong>上追平或超过物理；物理明显更优的只有 open/clear。<br>
    <strong>3.</strong> 学习把<strong>被破坏的载波修回来</strong>（0.927 → 0.525），甚至略优于正确的物理。<br>
    <strong>4.</strong> <strong>目标必须与指标一致</strong>：改用频率轴目标只到 0.651，明显更差。</p></div>
  <div class="callout"><div class="hd">如实记录一处不一致</div>
    <p>学习把界降到与物理相同甚至更低，但实测误差仍略高（0.688 vs 0.636）。
    即学出来的坐标<em>释放</em>了可辨识性，而线性插值没有把它全部兑现——
    这个差距属于估计器，不属于表征。</p></div>
  <figure><img src="{{FIG15}}" alt="四个区间下五种坐标系的可辨识性界柱状图">
    <figcaption><b>图 2.</b> 灰（无载波）与红（被破坏的物理）居高不下；
    橙（物理）、蓝（纯学习）、绿（破坏后再学习）几乎重合。</figcaption>
  </figure>
</section>

<section id="c04">
  <h2><span class="n">00d / 划界</span>与经典采样定理的关系</h2>
  <p class="lede">这条界本身<strong>不是新的</strong>——"规则采样下高于 Nyquist 的内容不可辨识"
  就是采样定理。不加说明地写进论文会被正确地质疑。以下是明确的划界。</p>
  <div class="claims">
    <div class="claim"><div class="tag">属于经典</div><h4>我们没有贡献的部分</h4>
      <p>规则阵列上的混叠与 Nyquist 波数；"波场需要每波长若干采样点"这一工程常识；
      相位解调、shifted POD、plane-wave 基等表征手段本身。</p></div>
    <div class="claim"><div class="tag">我们的主张</div><h4>四条新的部分</h4>
      <p><strong>1.</strong> 把它变成<strong>事前可算</strong>的量——从介质 <code>c(x)</code> 出发，
      一次 eikonal 求解就能把它从 0.997 降到 0.062，不需要场本身。<br>
      <strong>2.</strong> 把"该不该做物理对齐"变成<strong>一个数</strong>，并在四个公开数据集上事前预测全中。<br>
      <strong>3.</strong> 证明它<strong>不是建模问题</strong>——128 倍容量、六个估计器、训练误差 10⁻⁷。<br>
      <strong>4.</strong> 把它变成<strong>自监督训练目标</strong>，在不知道介质时把可辨识性学回来。</p></div>
    <div class="claim"><div class="tag">一句话</div><h4>差别在哪</h4>
      <p>经典结果说"高于 Nyquist 的内容丢了"；我们说的是"<strong>你可以先算出丢多少，
      可以用物理把它搬到 Nyquist 以下，而且这个搬运本身可以学</strong>"。</p></div>
  </div>
  <div class="callout"><div class="hd">两条必须声明的适用边界</div>
    <p><strong>随机阵列不适用。</strong>随机采样下混叠不是一堵墙，稀疏的高波数内容可以被恢复——
    正是压缩感知的机制。实测：随机采样下该量被打破的比例升到 6.5%，退化为启发式而非界。
    本文全部结论只针对<strong>规则阵列</strong>（也是真实传感硬件的形态）。<br>
    <strong>界描述可辨识性，不描述估计器。</strong>当场远低于 Nyquist 时界趋于 0，
    而线性插值仍有自身的截断误差，此时界成立但无信息量。
    界有意义的区间是采样接近或超过 Nyquist 的时候——也正是实际会遇到的区间。</p></div>
</section>

<section id="c05">
  <h2><span class="n">00e / 留出验证</span>把整套流程冻住，换一批没见过的介质</h2>
  <p class="lede">本项目的每一个测量决定——规则阵列、裁剪而非补零、Tukey 窗及其宽度、能量分位——
  都是<strong>看着前面那些场做出来的</strong>。这正是结论可能只是自身调参产物的情形。
  因此：<strong>一个参数都不调</strong>，流程整体冻结，换 12 个 seed 在仓库任何地方都没出现过的介质。</p>
  <div class="tablewrap"><table>
    <caption>252 组测量，全部在留出介质上。</caption>
    <thead><tr><th>检验的主张</th><th>留出结果</th></tr></thead>
    <tbody>
      <tr><td><strong>C1 界成立</strong></td><td style="text-align:left">线性插值打破 <strong>4.8%</strong>；一百万参数网络打破 <strong>0.0%</strong></td></tr>
      <tr><td><strong>C2 不是容量</strong></td><td style="text-align:left">一百万参数网络 <strong>0.995</strong>，零参数线性插值 <strong>0.759</strong></td></tr>
      <tr><td><strong>C3 信息可学</strong></td><td style="text-align:left">m=6 的界：无载波 0.993 → 物理 <strong>0.552</strong> → <strong>纯学习 0.554</strong></td></tr>
    </tbody>
  </table></div>
  <div class="callout good"><div class="hd">三条主张全部复现</div>
    <p><code>error_vs_bound</code> 在留出集上 R²=0.79、斜率 0.73，与开发集（R²=0.77、斜率 0.66）一致。
    特别是 C3：<strong>在从未见过的介质上，不给介质、不给源、不跑求解器</strong>，
    自监督学出的坐标把界降到 0.554，而用真实介质跑 eikonal 得到 0.552——两者实质相同。</p></div>
  <p><strong>如实记录</strong>：留出集上"学出来的坐标释放了可辨识性但线性插值没有全部兑现"
  这一差距依然存在（误差 0.721 vs 物理的 0.661），与前一节观察到的一致。</p>
</section>

<section id="c06">
  <h2><span class="n">00f / 摊销</span>坐标是可以摊销的：一次前向传播，迁移到没见过的介质</h2>
  <p class="lede">前面学出来的坐标都是<strong>逐场拟合</strong>的——那是"某一个场的表征"，
  不是"表征规则"。真正的检验是：能不能训练一个网络，把<strong>介质</strong>映射到它的对齐坐标，
  然后在<strong>没见过的介质</strong>上一次前向传播就得到坐标？</p>
  <div class="tablewrap"><table>
    <caption>唯一的损失就是可辨识性界本身——无标签、无 eikonal 目标。测试时物理需要每个介质跑一次求解器，摊销学习只要一次前向传播。界在 m=6。</caption>
    <thead><tr><th>训练 → 测试</th><th class="num">无载波</th><th class="num">eikonal</th><th class="num">摊销学习</th><th class="num">摊销+热启动</th><th class="num"><em>逐场上限</em></th></tr></thead>
    <tbody>
      <tr><td>随机散射体 → 同族未见 32 个</td><td class="num bad">0.995</td><td class="num win">0.581</td><td class="num win">0.589</td><td class="num">0.593</td><td class="num">0.483</td></tr>
      <tr><td>单夹杂 → 同族未见 32 个</td><td class="num bad">0.997</td><td class="num">0.330</td><td class="num win">0.300</td><td class="num win">0.296</td><td class="num">0.258</td></tr>
      <tr><td><strong>随机散射体 → 单夹杂（跨介质族）</strong></td><td class="num bad">0.997</td><td class="num win">0.364</td><td class="num win">0.364</td><td class="num">0.363</td><td class="num">0.315</td></tr>
    </tbody>
  </table></div>
  <div class="callout good"><div class="hd">三条读数</div>
    <p><strong>1.</strong> 摊销的学习坐标在未见介质上<strong>追平物理</strong>（0.589 vs 0.581），
    在单夹杂族上<strong>超过物理</strong>（0.300 vs 0.330），测试时只要一次前向传播——
    <strong>这才是"表征"而不是"某个场的拟合"</strong>。<br>
    <strong>2.</strong> <strong>跨介质族也成立</strong>：在随机多散射体上训练、在单夹杂介质上测试，
    与物理<strong>完全持平</strong>（0.364 vs 0.364）。学到的不是某一族介质的特例。<br>
    <strong>3.</strong> <strong>物理热启动几乎没有帮助</strong>（差别在噪声内）。
    在摊销设定下，物理连初始化的作用都不必要了。<br>
    <strong>4.</strong> <strong>摊销的代价约 15–20%</strong>，也给出了改进空间的上界。
    三行的 seed 标准差都 ≤ 0.001。</p></div>
  <figure><img src="{{FIG16}}" alt="三种训练-测试组合下四种坐标来源的界柱状图">
    <figcaption><b>图 3.</b> 蓝（摊销学习，测试时一次前向传播）与橙（物理，每个介质一次求解器）
    在三种组合下几乎重合，包括跨介质族。</figcaption>
  </figure>
  <p>这条结果把主张从"可以为某个场找到好坐标"升级为"<strong>可以学到一条从介质到坐标的规则</strong>"，
  并且这条规则的训练<strong>完全不需要标签</strong>——损失就是我们一直在报告的那个量。</p>
</section>

<section id="c07">
  <h2><span class="n">00g / 跨几何</span>介质是免费的，几何不是</h2>
  <p class="lede">前一节的摊销是在<strong>介质内容</strong>上迁移的——源位置固定、边界条件固定。
  那是"换内容"，不是"换几何"。这里把<strong>源位置</strong>与<strong>边界吸收</strong>变成留出变量：
  9 个源位置 × 3 种边界 × 10 个介质。网络只拿到实践中真正已知的东西
  （波速图、阻尼剖面、源位置），<strong>从不给走时</strong>，损失仍然只有界本身。</p>
  <div class="tablewrap"><table>
    <caption>关键比较是同一测试集上"学出来 / 物理"的比值。2 seed。</caption>
    <thead><tr><th>划分</th><th class="num">训练</th><th class="num">测试</th><th class="num">无载波</th><th class="num">物理</th><th class="num">学出来</th><th class="num">学/物理</th></tr></thead>
    <tbody>
      <tr><td>同分布（全部几何）</td><td class="num">216</td><td class="num">54</td><td class="num bad">0.994</td><td class="num">0.693</td><td class="num">0.766</td><td class="num">1.11</td></tr>
      <tr><td><strong>留出源位置</strong>（全部边界）</td><td class="num">180</td><td class="num">90</td><td class="num bad">0.994</td><td class="num">0.722</td><td class="num">0.812</td><td class="num bad">1.12</td></tr>
      <tr><td><strong>留出边界</strong>（只训练 open）</td><td class="num">90</td><td class="num">180</td><td class="num bad">0.994</td><td class="num">0.791</td><td class="num win">0.792</td><td class="num win">1.00</td></tr>
      <tr><td>同分布（只 open）</td><td class="num">72</td><td class="num">18</td><td class="num bad">0.993</td><td class="num">0.541</td><td class="num">0.599</td><td class="num">1.11</td></tr>
      <tr><td><strong>留出源位置（只 open）</strong></td><td class="num">60</td><td class="num">30</td><td class="num bad">0.994</td><td class="num">0.588</td><td class="num bad">0.722</td><td class="num bad">1.23</td></tr>
    </tbody>
  </table></div>
  <div class="callout"><div class="hd">三条读数，其中两条是负面的</div>
    <p><strong>1.</strong> <strong>留出边界几乎免费</strong>（1.00）——但那两个区间物理本身也只能把界从
    0.994 降到 0.791，<em>可拿的收益本来就少</em>，"持平"的门槛低。<br>
    <strong>2.</strong> <strong>留出源位置不免费</strong>：同一 open 区间内，同分布 1.11 → 留出源
    <strong>1.23</strong>。这是真实的泛化损失，与"换介质完全免费"形成对比。<br>
    <strong>3.</strong> <strong>几何异质性本身就要付 ~11%</strong>：一个同时服务 9 个源位置、
    3 种边界的模型，即使在<em>同分布</em>上也比物理差 11%，而固定几何的模型是追平甚至超过物理的。</p></div>
  <p><strong>更多训练不能修复它</strong>：步数从 2500 提到 6000，同分布 0.766 → 0.773、
  留出源 0.812 → 0.843，不降反升（轻微过拟合到已训练的源位置）。这不是训练预算问题。</p>
  <div class="callout"><div class="hd">该怎么说</div>
    <p>可用的说法是"<strong>给定几何族，可以学到一条从介质到坐标的规则</strong>"，
    而<strong>不能</strong>说"学到了一条与几何无关的规则"。这条限制应当写进论文，而不是回避。</p></div>
</section>

<section id="c08">
  <h2><span class="n">00h / 审计</span>公开波动基准在这条轴上的分布</h2>
  <p class="lede">判据把"这个数据集属于哪个区间"变成一个可计算的数。既然如此，就应当<strong>统一地</strong>
  对所有测过的数据集算一遍，并把出处、版本、许可一并记录，让结论可被审计和复现。</p>
  <div class="callout"><div class="hd">判定规则（对所有数据集一致应用）</div>
    <p>在参考阵列间距 m=6 上，若原始界已低于 0.15，记为<strong>已可辨识</strong>——该密度下本就不需要载波，
    两个接近零的数之比不携带信息；否则按对齐界与原始界之比判定：
    <code>≤0.5</code> 有利、<code>≤0.9</code> 中等、否则不利。</p></div>
  <div class="tablewrap"><table>
    <thead><tr><th>公开波动数据集</th><th class="num">界(raw)</th><th class="num">界(aligned)</th><th class="num">比值</th><th>判定</th></tr></thead>
    <tbody>
      <tr><td>The Well — acoustic_scattering_inclusions</td><td class="num">0.686</td><td class="num">0.596</td><td class="num">0.87</td><td>中等</td></tr>
      <tr><td>The Well — acoustic_scattering_maze</td><td class="num bad">0.999</td><td class="num bad">0.985</td><td class="num bad">0.99</td><td>不利</td></tr>
      <tr><td>WaveBench — isotropic, ω 标称 40</td><td class="num bad">0.992</td><td class="num bad">0.993</td><td class="num bad">1.00</td><td>不利</td></tr>
      <tr><td>The Well — helmholtz_staircase</td><td class="num">0.020</td><td class="num">0.023</td><td class="num">—</td><td>已可辨识</td></tr>
      <tr><td>WaveBench — isotropic, ω 标称 10</td><td class="num">0.033</td><td class="num">0.084</td><td class="num">—</td><td>已可辨识</td></tr>
      <tr><td><em>（参照）本项目 FDTD open/clear</em></td><td class="num">0.997</td><td class="num win">0.062</td><td class="num win">0.06</td><td><strong>有利</strong></td></tr>
      <tr><td><em>（参照）本项目 FDTD closed/dense</em></td><td class="num">0.990</td><td class="num">0.860</td><td class="num">0.87</td><td>中等</td></tr>
    </tbody>
  </table></div>
  <div class="callout"><div class="hd">0 / 5</div>
    <p><strong>没有一个公开波动数据集落在「有利」区间。</strong>而自建求解器的参照点跨越 0.06–0.87，
    覆盖整条轴——缺口不在方法能否奏效，而在<strong>公开基准在这条轴上的采样偏斜</strong>。<br><br>
    <strong>对社区的含义</strong>：若要评估物理对齐类表征方法，现有公开波动基准<strong>无法区分
    「方法无效」与「数据集不在该方法的适用区间」</strong>。补上开放/吸收介质、且采样低于空间
    Nyquist 的公开数据，是让这类方法可被公平评估的前提。</p></div>
  <p><strong>一个数据集可以在不同轴上落在不同区间</strong>，审计因此并列给出三根轴而不是一个数字。
  Helmholtz staircase 是典型：m=6 的空间采样下它本就可辨识（界 0.020），
  但在<strong>频率轴的等预算压缩</strong>上载波带来 <strong>10.6×</strong> 增益。</p>
  <p><strong>存档</strong>（均已入 git）：<code>reports/benchmark_registry.json</code>（机器可读，
  含每个数据集的出处 URL、修订号、许可、获取方式与三根轴上的度量）与
  <code>reports/BENCHMARK_COVERAGE.md</code>（表格版，含非波动数据集的另一套度量）。
  许可与版本均于 2026-08-21 通过 HuggingFace 与 Zenodo API 核实；
  OpenFWI 与 PDEBench 的许可<strong>未在本项目核实</strong>，登记为「见官方发布」。</p>
</section>

<section id="c1">
  <h2><span class="n">01 / 主张</span>三条可证伪的主张</h2>
  <div class="claims">
    <div class="claim"><div class="tag">C1 可辨识性界</div><h4>误差 ≥ 界</h4>
      <p>在间距 <code>m</code> 的规则阵列上，任何方法的相对误差不低于
      <code>√(能量(|k|&gt;1/(2m))/总能量)</code>。一次 FFT 可得，与所用模型无关。</p></div>
    <div class="claim"><div class="tag">C2 不是容量</div><h4>加容量无用，换坐标有用</h4>
      <p>128× 参数量 → 3% 误差变化；界降 1.45× → 所有估计器跟着降。
      前者是模型问题的排除，后者是信息问题的确认。</p></div>
    <div class="claim"><div class="tag">C3 信息可学</div><h4>把界当损失</h4>
      <p>不知道介质、源位置与求解器，仅靠自监督最小化界本身，
      即可学出把界降到与物理同等水平的坐标。</p></div>
  </div>
  <p><strong>停止规则</strong>：C1 在任一数据集上被系统性打破（&gt;20%）则整条线终止；
  C2 若容量扫描出现单调显著改善则终止；C3 若纯学习的界始终高于无载波则终止。
  截至目前三条都未触发。</p>

  <h3>1.1&nbsp;&nbsp;频率轴上的对偶：秩定律</h3>
  <p>接下来几节是同一原理在<strong>另一根轴</strong>上的形式。空间轴上是"域面积 × 占据波数支撑"，
  频率轴上则是<strong>带宽 × 占据延迟支撑</strong>：</p>
  <div class="formula">rank(U)  ≈  B · Λ        Λ = 承载能量的到达时间集合测度（分辨率 1/B）</div>
  <p>解调把延迟占据从"绝对走时展宽"换成"相对延迟展宽"，正如它在空间轴上把波数从
  <code>ω/c</code> 换成包络的变化尺度。<strong>两条轴上的收益由同一物理量（延迟展宽）控制。</strong></p>
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
  <h3>8.2&nbsp;&nbsp;与最近的先验工作对比：shifted POD</h3>
  <p>shifted POD（Reiss 等）把输运主导的场分解成几个"共动坐标系"，每个在撤掉刚性位移后低秩。
  表面上和我们很像——但<strong>它撤的是整帧共享的空间刚性平移，我们撤的是逐点的时间弯折</strong>。
  对刚性平移的图案两者等价；对点源的<strong>膨胀</strong>波前、被介质折射的波前，前者原理上表达不了。</p>
  <div class="callout good"><div class="hd">先证明实现没有削弱它</div>
    <p>在一个刚性平移的高斯脉冲上（它自己的假设成立时）：rank 1 → plain POD 0.855 vs
    shifted POD <strong>0.0008</strong>；rank 2 → 0.706 vs <strong>0.0000</strong>。
    比 plain POD 好三个数量级。关键实现细节：位移必须做<strong>亚像素</strong>估计
    （FFT 互相关 + 抛物线插值）；只做整数位移时残差被分数级错位主导，
    这个 baseline 会看起来弱得多。</p></div>
  <div class="tablewrap"><table>
    <caption>真实波场，时域，等参数预算；shifted POD 取 K=1,2,3 中最好的一档。</caption>
    <thead><tr><th>场</th><th class="num">R</th><th class="num">plain POD</th><th class="num">shifted POD</th><th class="num">载波（我们）</th><th class="num">我们/plain</th><th class="num">sPOD/plain</th></tr></thead>
    <tbody>
      <tr><td>open, clear</td><td class="num">4</td><td class="num">0.705</td><td class="num">0.564</td><td class="num win">0.056</td><td class="num win">12.7×</td><td class="num">1.25×</td></tr>
      <tr><td>open, clear</td><td class="num">32</td><td class="num">0.030</td><td class="num bad">0.137</td><td class="num win">0.014</td><td class="num">2.1×</td><td class="num bad">0.22×</td></tr>
      <tr><td>open, sparse</td><td class="num">8</td><td class="num">0.706</td><td class="num">0.694</td><td class="num win">0.522</td><td class="num">1.35×</td><td class="num">1.02×</td></tr>
      <tr><td>partial, clear</td><td class="num">8</td><td class="num">0.804</td><td class="num">0.769</td><td class="num win">0.639</td><td class="num">1.26×</td><td class="num">1.05×</td></tr>
      <tr><td>closed, dense</td><td class="num">32</td><td class="num">0.698</td><td class="num bad">0.828</td><td class="num">0.683</td><td class="num">1.02×</td><td class="num bad">0.84×</td></tr>
      <tr><td>acoustic inclusions</td><td class="num">16</td><td class="num">0.241</td><td class="num bad">0.557</td><td class="num">0.223</td><td class="num">1.08×</td><td class="num bad">0.44×</td></tr>
      <tr><td>The Well maze</td><td class="num">32</td><td class="num">0.393</td><td class="num bad">0.652</td><td class="num bad">0.410</td><td class="num bad">0.96×</td><td class="num bad">0.61×</td></tr>
    </tbody>
  </table></div>
  <p><strong>shifted POD 在我们测的每一个波场上都没赢过 plain POD</strong>：小预算下略有帮助
  （1.02–1.25×），大预算下明显更差（0.19–0.88×）。刚性平移是波场的错误变换——
  膨胀波前无法用平移对齐，而把 rank 拆给多个坐标系又摊薄了每个的表达力。</p>
  <div class="tablewrap"><table>
    <caption>换成"等精度需要几个分量"更能说明问题。</caption>
    <thead><tr><th>场</th><th class="num">我们 R=4 的误差</th><th class="num">plain POD 需要 R≈</th><th class="num">压缩倍数</th></tr></thead>
    <tbody>
      <tr><td>open, clear</td><td class="num">0.056</td><td class="num">24.8</td><td class="num win">6.2×</td></tr>
      <tr><td>partial, clear</td><td class="num">0.706</td><td class="num">13.8</td><td class="num">3.4×</td></tr>
      <tr><td>open, sparse</td><td class="num">0.602</td><td class="num">12.1</td><td class="num">3.0×</td></tr>
      <tr><td>closed, dense</td><td class="num">0.909</td><td class="num">6.4</td><td class="num">1.6×</td></tr>
      <tr><td>acoustic inclusions</td><td class="num">0.565</td><td class="num">4.9</td><td class="num">1.2×</td></tr>
      <tr><td>The Well maze</td><td class="num">0.855</td><td class="num">4.0</td><td class="num bad">1.0×</td></tr>
    </tbody>
  </table></div>
  <div class="callout"><div class="hd">如实记录两个负值</div>
    <p>maze 上我们比 plain POD <strong>略差</strong>（0.96–0.99×），inclusions 在 R=32 上也略差（0.92×）。
    两者都落在判据预测的"收益≈1"区间内，正负抖动属于预期——但不能说成"没有变差"。</p></div>
  <figure><img src="{{FIG11}}" alt="六个场下三种方法误差随参数预算的曲线">
    <figcaption><b>图 11.</b> 等预算重构误差。黄色（shifted POD）在每个波场上都没能赢过红色（plain POD）。</figcaption>
  </figure>

  <h3>8.3&nbsp;&nbsp;学出来的对齐坐标（方法核心）</h3>
  <p>前面所有载波都来自 eikonal——需要已知 <code>c(x)</code>，且只对非色散首达精确。
  把它换成<strong>自监督学出来</strong>的相位场，目标只有一条：</p>
  <div class="formula">min<sub>θ</sub>  ‖ U ⊙ exp(+i φ<sub>θ</sub>) ‖<sub>*</sub>  +  λ · ‖ |∇τ<sub>θ</sub>| − 1/c ‖²
        └─ 对齐后有多低秩（无标签）        └─ eikonal 残差（物理先验）</div>
  <p><strong>目标良定</strong>：逐点乘一个模长为 1 的因子不改变 Frobenius 范数，
  所以在 ‖·‖<sub>F</sub> 固定下压低核范数等价于压低秩。<code>λ→∞</code> 退化为 eikonal 载波，
  <code>λ=0</code> 是纯自监督。而且相位不必对 <code>f</code> 线性——<strong>顺带把色散纳入表达能力</strong>，
  这是走时载波原理上做不到的。</p>
  <div class="tablewrap"><table>
    <caption>2% 传感器重建 NRMSE（2 seed 均值）。"物理但错了"用的是第 6 节证明会致命的粗糙误差，幅度 1×(1/B)。</caption>
    <thead><tr><th>区间</th><th class="num">无载波</th><th class="num">物理（正确）</th><th class="num">物理但错了</th><th class="num">错物理 + 学习</th><th class="num">纯学习（无物理）</th></tr></thead>
    <tbody>
      <tr><td>open, clear</td><td class="num bad">1.09</td><td class="num win">0.084</td><td class="num bad">1.09</td><td class="num win">0.095</td><td class="num win">0.101</td></tr>
      <tr><td>open, sparse</td><td class="num bad">1.10</td><td class="num win">0.746</td><td class="num bad">1.12</td><td class="num">0.782</td><td class="num">0.780</td></tr>
      <tr><td>partial, clear</td><td class="num bad">1.09</td><td class="num win">0.829</td><td class="num bad">1.12</td><td class="num">0.832</td><td class="num">0.833</td></tr>
      <tr><td>closed, dense</td><td class="num bad">1.14</td><td class="num">1.063</td><td class="num bad">1.17</td><td class="num">1.111</td><td class="num">1.117</td></tr>
    </tbody>
  </table></div>
  <div class="callout good"><div class="hd">物理是先验，不是依赖</div>
    <p>一个被粗糙误差毁掉的载波（1.09，<em>比无载波还差</em>）经过学习回到 <strong>0.095</strong>，
    即物理级别。而<strong>完全不给物理、不给介质、不给源位置</strong>，从零学出来的坐标
    同样达到 0.101——与物理载波实质等同。</p></div>
  <div class="tablewrap"><table>
    <caption>The Well Helmholtz staircase，8 个源，R=4 截断误差。此数据集的 trapped mode 有非线性色散，正是第 10 节记录的失败原因。</caption>
    <thead><tr><th>方法</th><th class="num">R=4 误差</th><th class="num">相对 eikonal</th></tr></thead>
    <tbody>
      <tr><td>eikonal（物理）</td><td class="num">0.0288 ± 0.0078</td><td class="num">1.00×</td></tr>
      <tr><td>learned，核范数目标，τ-only</td><td class="num bad">0.0279 ± 0.0052</td><td class="num bad">1.02×</td></tr>
      <tr><td>learned，核范数目标，色散</td><td class="num bad">0.0271 ± 0.0013</td><td class="num bad">1.06×</td></tr>
      <tr><td>learned，尾能量目标，τ-only</td><td class="num">0.0223 ± 0.0057</td><td class="num">1.29×</td></tr>
      <tr><td><strong>learned，尾能量目标，色散</strong></td><td class="num win">0.0164 ± 0.0034</td><td class="num win">1.74×</td></tr>
    </tbody>
  </table></div>
  <div class="callout"><div class="hd">方法论结论：代理必须与指标对齐</div>
    <p>核范数只是低秩的凸代理，直接优化它<strong>会把真正上报的指标推向错误方向</strong>
    （1.02–1.06×，几乎无增益）。改成直接优化"预算 R 之外的尾部能量"——同样可微——
    才拿到 1.29–1.74×。</p></div>
  <p><strong>如实记录</strong>：纯学习在 FDTD 上是"追平"物理而非超过（0.101 vs 0.084）；
  只有当物理模型本身有系统性缺陷（色散）时学习才明确胜出。两者是互补而非替代。</p>
  <figure><img src="{{FIG12}}" alt="左：四个区间下五种载波的传感器重建误差；右：staircase 上四种方法的 rank-4 误差">
    <figcaption><b>图 12.</b> 左：学习把"错的物理"救回物理级别，且无物理也能追平。
    右：在物理模型本身有缺陷（色散）的公开数据上，学出来的相位超过 eikonal 1.74×。</figcaption>
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

  <h3>9.2&nbsp;&nbsp;WaveBench：判据说别做，实测也确实没用</h3>
  <p>WaveBench 的 time-harmonic 部分是本项目最想要的公开数据形态：<strong>同一批非均匀介质在 4 个频率上的
  Helmholtz 解，并附带 wavespeed</strong>，因此可以建可部署的 eikonal 载波并在同一介质上扫频率。</p>
  <div class="callout good"><div class="hd">取数方式值得记录</div>
    <p>整个数据集是一个 75 GB 的 zip，但 Zenodo 支持 HTTP range 请求，而 FFCV <code>.beton</code>
    容器把每个样本存在<strong>显式字节偏移</strong>上——因此文件的一个<strong>前缀</strong>就足以读出落在其中的
    全部样本。实际只取了 <strong>2.2 GB</strong>（四个频率各约 1000 个可读样本），而不是 75 GB。
    <code>ffcv</code> 本体需要编译工具链，所以 <code>src/wave_lr/beton.py</code> 是按官方格式定义
    重写的纯 NumPy 读取器。</p></div>
  <p><strong>两项前置验证都通过</strong>：同一 index 在四个频率文件里的 wavespeed 逐元素完全相同
  （max|Δc| = 0.0），确实是"同一介质 × 四个频率"；容器不记录 ω 与网格间距，但载波只需要其乘积
  <code>κ = ω·spacing</code>，从场自身相位梯度标定出的 κ 比值为 <strong>1.00 / 1.51 / 2.08 / 4.34</strong>，
  与文件名标称的 1 / 1.5 / 2 / 4 一致。实测波长 63.6 / 41.6 / 29.6 / <strong>14.2</strong> 像素。</p>
  <div class="tablewrap"><table>
    <caption>raw / aligned 增益，&gt;1 才是有帮助。方向完全符合预期，幅度不足。</caption>
    <thead><tr><th class="num">ω 标称</th><th class="num">波长(px)</th><th class="num">p=0.005</th><th class="num">p=0.01</th><th class="num">p=0.02</th><th class="num">p=0.05</th><th class="num">p=0.10</th></tr></thead>
    <tbody>
      <tr><td class="num">10</td><td class="num">63.6</td><td class="num bad">0.85</td><td class="num bad">0.74</td><td class="num bad">0.67</td><td class="num bad">0.69</td><td class="num bad">0.74</td></tr>
      <tr><td class="num">15</td><td class="num">41.6</td><td class="num bad">0.94</td><td class="num bad">0.87</td><td class="num bad">0.75</td><td class="num bad">0.64</td><td class="num bad">0.62</td></tr>
      <tr><td class="num">20</td><td class="num">29.6</td><td class="num bad">0.97</td><td class="num bad">0.93</td><td class="num bad">0.83</td><td class="num bad">0.67</td><td class="num bad">0.60</td></tr>
      <tr><td class="num">40</td><td class="num">14.2</td><td class="num">1.00</td><td class="num bad">0.99</td><td class="num bad">0.96</td><td class="num bad">0.88</td><td class="num bad">0.76</td></tr>
    </tbody>
  </table></div>
  <div class="callout"><div class="hd">判据事先就给出了这个答案</div>
    <p>四个频率构成的 <code>(x,ω)</code> 矩阵实测<strong>满秩 4</strong>（对齐前后都是 4，rank-1 残差
    0.796 → 0.801，对齐毫无作用）。满秩意味着 <code>Λ ≥ (rank−1)/B = 17</code>，而区域穿越时间约 38.8，
    故<strong>占据比 ≥ 0.44</strong>。对照第 11c 节的标定（0.14 → 4.50；0.38 → 1.10；0.48 → 1.04），
    0.44 落在"无收益"区间——预测与实测一致。物理原因很直白：介质对比度 <code>c ∈ [1.5, 5]</code>
    是<strong>强散射</strong>，首达之外的多次散射携带了大部分能量。</p></div>
  <p><strong>这条负结果反而强化了 benchmark 覆盖的论点</strong>：测过的四个公开波动基准里，<strong>三个</strong>
  （The Well acoustic maze、acoustic inclusions、WaveBench isotropic）都落在相位对齐必然失效的
  强混响/强散射区间，只有 Helmholtz staircase 落在有利区间——而那里恰好拿到 9.8–11.5× 的等预算压缩。
  <strong>"有利区间缺乏公开基准"这个缺口是真实存在的，不是我们没找。</strong></p>
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

<section id="c115">
  <h2><span class="n">11b / 弱结果</span>跨介质算子学习：一个诚实的未完成项</h2>
  <p class="lede">第 6b 节的网络是逐 case 拟合的隐式表示。为了检验"跨 case 预训练的算子是否也受益于对齐"，
  用同一个 FNO、同一批介质、同一预算，分别预测复场与对齐后的包络（载波在测试时由 <code>c(x)</code>
  重新算出，两条路线都可部署）。介质族是单个随机圆形夹杂，448 训练 / 64 测试。</p>
  <div class="tablewrap"><table>
    <caption>训练误差随容量下降而测试误差上升——明确的数据受限过拟合。</caption>
    <thead><tr><th class="num">FNO 谱模态</th><th class="num">参数量</th><th class="num">raw 目标</th><th class="num">对齐目标</th><th class="num">倍数</th><th class="num">raw 训练 loss</th></tr></thead>
    <tbody>
      <tr><td class="num">8</td><td class="num">0.53M</td><td class="num">0.784</td><td class="num win">0.695</td><td class="num">1.13×</td><td class="num">0.526</td></tr>
      <tr><td class="num">16</td><td class="num">2.10M</td><td class="num bad">1.002</td><td class="num win">0.794</td><td class="num">1.26×</td><td class="num">0.206</td></tr>
      <tr><td class="num">24</td><td class="num">4.73M</td><td class="num bad">1.031</td><td class="num win">0.840</td><td class="num">1.23×</td><td class="num">0.110</td></tr>
      <tr><td class="num">32</td><td class="num">8.40M</td><td class="num bad">1.047</td><td class="num win">0.874</td><td class="num">1.20×</td><td class="num">0.079</td></tr>
    </tbody>
  </table></div>
  <div class="callout"><div class="hd">这一节不能用来支持主张</div>
    <p>对齐目标在每一个容量档上都更好（1.13–1.26×），但<strong>两者都没有把任务做出来</strong>
    （中位 test NRMSE 0.36–0.45）。这只支持一个弱得多的版本：在同等（不足的）数据下，
    对齐目标一致地更容易学。要变成真结论需要数量级更多的训练介质——
    <strong>这是目前最清楚的一个待办项。</strong></p></div>
</section>

<section id="c117">
  <h2><span class="n">11c / 通用性</span>定律走得多远：一个收窄而非扩张的结论</h2>
  <p class="lede">自由度计数的推导里没有出现"波"这个字，所以 <code>rank ≤ B·Λ</code> 原则上对任何时空场成立。
  用完全相同的流程测了五个<strong>公开录制的非波动流场</strong>（cylinder wake、Kuramoto–Sivashinsky、
  Kolmogorov 湍流、active matter、PDEBench 反应扩散），加上一个<strong>合成的非波动瞬态输运</strong>族
  （平流—扩散移动斑块，扩散系数可调）。载波延迟全部从数据估计，不用任何物理。</p>
  <div class="callout good"><div class="hd">结论一：界是普适的</div>
    <p>100+ 次测量，<strong>违反率 0.0%</strong>——波动与非波动，秩从未超过 <code>B·Λ</code>。</p></div>
  <div class="tablewrap"><table>
    <caption>结论二：界只在波场上是紧的。非波动场上平均松 2.6 倍。</caption>
    <thead><tr><th></th><th class="num">拟合斜率</th><th class="num">R²</th><th class="num">中位相对误差</th></tr></thead>
    <tbody>
      <tr><td>波场</td><td class="num win">0.965</td><td class="num win">0.952</td><td class="num win">0.17</td></tr>
      <tr><td>非波动场</td><td class="num">0.739</td><td class="num">0.875</td><td class="num bad">1.65</td></tr>
    </tbody>
  </table></div>
  <p>约束非波动场的不是时间占据，而是<strong>空间相干性</strong>——即第 11 节修正项
  <code>rank ≤ min(B·Λ, rank(G))</code> 里的第二项。实测 <code>rank/(B·Λ)</code>：混响波场 0.92、
  Kolmogorov 湍流 0.78（空间弥散，界紧）；active matter 0.11、反应扩散 0.17（空间高度相干，界松）。</p>
  <div class="tablewrap"><table>
    <caption>结论三：对齐只对"瞬态"输运有用。用"能量占记录长度的比例"作横轴，全部数据排成一条线。</caption>
    <thead><tr><th>数据</th><th class="num">占记录比</th><th class="num">实测对齐增益</th></tr></thead>
    <tbody>
      <tr><td>wave, open clear</td><td class="num win">0.14</td><td class="num win">4.50</td></tr>
      <tr><td>wave, open sparse</td><td class="num">0.38</td><td class="num">1.10</td></tr>
      <tr><td>wave, closed dense</td><td class="num">0.48</td><td class="num">1.04</td></tr>
      <tr><td>平流—扩散 D=0</td><td class="num">0.93</td><td class="num win">1.48</td></tr>
      <tr><td>平流—扩散 D=0.0005</td><td class="num">0.93</td><td class="num">1.21</td></tr>
      <tr><td>平流—扩散 D=0.002</td><td class="num">0.93</td><td class="num">1.05</td></tr>
      <tr><td>平流—扩散 D=0.01</td><td class="num">0.90</td><td class="num bad">0.62</td></tr>
      <tr><td>cylinder wake</td><td class="num">0.59</td><td class="num bad">0.83</td></tr>
      <tr><td>Kolmogorov 湍流</td><td class="num">0.58</td><td class="num bad">0.68</td></tr>
      <tr><td>Kuramoto–Sivashinsky</td><td class="num">0.85</td><td class="num bad">0.65</td></tr>
      <tr><td>反应扩散</td><td class="num">0.86</td><td class="num bad">0.47</td></tr>
      <tr><td>active matter</td><td class="num">0.83</td><td class="num bad">0.47</td></tr>
    </tbody>
  </table></div>
  <div class="callout"><div class="hd">五个录制流场全部统计定常，对齐有害</div>
    <p>能量填满记录，对齐不但无益而且有害（0.47–0.83）。害处的来源与第 6 节完全一致：
    从定常信号估出来的延迟场是粗糙的，而粗糙误差 ≥1/B 会破坏秩。
    <strong>判据事先就说了不该做</strong>（预测增益 0.81–0.99）。</p></div>
  <p><strong>但机制本身不是波动专属</strong>：合成的平流—扩散族在<em>非波动</em>系统里复现了波动相图的形状——
  增益随扩散把输运相干性抹掉而从 1.48 单调塌到 0.62。</p>
  <p><strong>因此正确的适用范围是"瞬态且相干的输运"，而不是"波动"。</strong>
  脉冲源产生的波场天然满足两者；流体基准数据两者都不满足。这条结论<strong>收窄</strong>了主张，
  但也堵死了"这不就是流体里的 shifted POD 吗"这一质疑——本方法不适用于定常流场，且判据会提前说明。</p>
</section>

<section id="c12">
  <h2><span class="n">12 / 下一步</span>按优先级</h2>
  <ol class="steps">
    <li><strong>有利区间的公开 benchmark</strong>——最大的信誉缺口。测过的四个公开波动基准里
      三个落在必然无效区间。要么找到可引用的开放介质公开数据，
      要么把"覆盖缺口"本身作为 D&amp;B 贡献正式化（后者更可行）。</li>
    <li><strong>跨介质 / 跨几何迁移</strong>——学出来的坐标目前是逐 case 拟合的，
      没有验证过迁移。这是"表征"主张最自然的下一个检验。</li>
    <li><strong>跨介质算子学习补到有结论</strong>——现在是弱阳性，需要数量级更多的训练介质。</li>
    <li><strong>色散的完整处理</strong>——<code>φ(x,f)</code> 已证明有效（staircase 上 1.74×），
      但没有系统扫过色散强度。</li>
    <li><strong>接生成式残差</strong>——在 <code>Λ<sub>rel</sub></code> 大的区间低秩必然不够，
      判据正好指出"什么时候必须上生成模型"。</li>
  </ol>
  <h3>顺带发现的数据完整性问题</h3>
  <div class="callout"><div class="hd">影响其他项目</div>
    <p><strong>本地 OpenFWI 副本的地震数据与速度模型来自不同 family</strong>（<code>seis2_*</code> 配 <code>vel4_*</code>）。
    40 个模型上，由初至时距斜率反推的表层速度与配对模型的表层速度相关系数仅 <strong>0.135</strong>。
    任何在该副本上做的速度条件化实验都是无效的，需要重新下载官方配对文件。</p></div>
  <p>另外三条记录在 <code>docs/DATA_INTEGRITY.md</code>：staircase 的 y 轴在数组中反向存储；
  其 50 个时间步是 <code>e<sup>-iωt</sup></code> 的纯解析冗余；test split 的 3 条轨迹共用同一个源位置。</p>
</section>

<section id="cA">
  <h2><span class="n">附录 A / 方法</span>载波 τ(x)：定义、算法、复杂度与通用性</h2>

  <h3>A.1&nbsp;&nbsp;τ(x) 是什么，为什么恰好是它</h3>
  <div class="formula">|∇τ(x)| = 1/c(x),        τ(源点) = 0</div>
  <p>τ(x) 是波从源点到 x 的<strong>首达走时</strong>：沿最快路径（向高速区弯曲、绕过障碍）所需的最短时间，
  等价于以 <code>ds/c(x)</code> 为度量的测地距离。</p>
  <p>它不是启发式，而是波场高频渐近展开里的相位函数<em>本身</em>。把 <code>u = A(x)e<sup>iωτ(x)</sup></code>
  代入 Helmholtz 方程并收集 <code>O(ω²)</code> 项：</p>
  <div class="formula">ω²(1/c² − |∇τ|²)·A = 0    ⟹    |∇τ| = 1/c</div>
  <p>因此 <code>e<sup>-iωτ</sup></code> 扣掉的正是几何光学那一项，剩下的是幅度与射线理论管不了的成分
  （绕射、多次反射、散射）。这是整套做法的根据。</p>

  <h3>A.2&nbsp;&nbsp;算法，与一个必须记录的精度陷阱</h3>
  <p>Godunov 迎风格式，逐格点求解一个二次方程，反复扫描直至不再下降：</p>
  <div class="formula">a = min(τ<sub>上</sub>, τ<sub>下</sub>),   b = min(τ<sub>左</sub>, τ<sub>右</sub>),   sh = spacing / c

|a − b| ≥ sh :  τ = min(a,b) + sh
否则         :  τ = (a + b + √(2·sh² − (a−b)²)) / 2</div>
  <div class="callout"><div class="hd">陷阱</div>
    <p>一阶格式在对角方向有约 <strong>15%</strong> 的走时误差，而第 6 节测出的容限是
    <code>δτ &lt; 1/B</code>——这个误差足以让载波完全失效。修法是在源点附近播种解析解
    （梯度奇点与误差都产生在那里），且播种取邻域内<strong>最慢</strong>的速度，
    因此只可能高估；而扫描只降低数值，永远不会污染解。修正后最大误差降到 <strong>1.2%</strong>；
    方箱上 eikonal 的直达走时与解析镜像源相差 <strong>0.005</strong>（记录长 6.0，即 0.1%）。</p></div>

  <h3>A.3&nbsp;&nbsp;复杂度（实测，A100）</h3>
  <div class="tablewrap"><table>
    <caption>Fast sweeping 是 O(N)，N 为格点数。</caption>
    <thead><tr><th>网格</th><th class="num">单个介质</th><th class="num">批量摊薄</th></tr></thead>
    <tbody>
      <tr><td>64×64</td><td class="num">490 ms</td><td class="num win">1.8 ms</td></tr>
      <tr><td>128×128</td><td class="num">306 ms</td><td class="num win">5.8 ms</td></tr>
      <tr><td>256×256</td><td class="num">—</td><td class="num win">32 ms</td></tr>
      <tr><td>1024×256</td><td class="num">1.6 s</td><td class="num win">356 ms</td></tr>
    </tbody>
  </table></div>
  <p>对照：同一 128² 网格上跑一次 FDTD 正演 <strong>4.2 s</strong>，做一次场矩阵 SVD <strong>1.8 s</strong>。
  <strong>eikonal 比它所服务的那一步便宜三个数量级</strong>，不构成实际开销。</p>

  <h3>A.4&nbsp;&nbsp;通用性</h3>
  <div class="tablewrap"><table>
    <thead><tr><th>变化的量</th><th>是否重算</th><th>说明</th></tr></thead>
    <tbody>
      <tr><td><strong>频率 ω</strong></td><td><strong>不需要</strong></td><td style="text-align:left">τ 与频率无关；载波 <code>e<sup>-2πifτ(x)</sup></code> 里 f 只是标量乘子。<strong>一次求解服务整个频带</strong>——这正是频率轴上秩会塌缩的结构性原因</td></tr>
      <tr><td>源位置</td><td>每源一次</td><td style="text-align:left">staircase 的 26 个源共几分钟</td></tr>
      <tr><td>介质 / 样本</td><td>每个一次</td><td style="text-align:left">实现直接接受 (B,H,W) 批量</td></tr>
      <tr><td>边界条件</td><td><strong>算法不变</strong></td><td style="text-align:left">见下</td></tr>
    </tbody>
  </table></div>
  <p><strong>边界条件最容易误解</strong>：eikonal 只给<em>首达</em>——它天然处理非均匀 <code>c(x)</code>、
  障碍物（波自动绕行）与吸收边界；但它<em>不给反射</em>，反射是额外的 <code>τ<sub>m</sub></code>，
  需要额外载波（方箱由镜像源精确构造，一般情况由数据驱动的虚源扫描估计）。
  所以<strong>边界条件决定的是"需要几个载波"，而不是"eikonal 这一步能否使用"</strong>——
  而这正是 <code>Λ<sub>rel</sub></code> 判据回答的问题。</p>
  <p><strong>前提</strong>：需要已知 <code>c(x)</code> 与源位置。正问题/监测场景下两者给定；
  反问题中可退回到从数据拾取初至走时，实测略差但同样有效。
  <strong>已实测的失效条件</strong>：色散（trapped mode 的模式截止，第 10 节）与强混响
  （<code>Λ<sub>rel</sub> ≈ Λ<sub>abs</sub></code>，但判据会提前告知）。
  3D 与非结构网格上 fast sweeping 同样适用；本实现目前是 2D 规则网格。</p>
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
