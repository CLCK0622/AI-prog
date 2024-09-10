#import "template.typ": *
#import "@preview/lovelace:0.3.0": *

#show: project.with(
  title: [基于 U-Net 架构的路网预测方案],
  abstract: [作者提出了一种创新的 U-Net 架构应用，通过 AdamW 优化器和 Focal Loss 的应用，本模型可以用于通过地理空间数据来预测可能的道路网络。该机器学习模型的训练使用了河流网络和住宅区数据，以及 TIFF 格式的高程数据。该模型通过使用带有 pos_weight 张量的 Focal Loss，使其对类别不平衡导致的在复杂的城市和农村景观中代表性较少的小型道路及偏远的居民区等少数信息更加敏感。模型在 60% 的训练集、20% 的验证集和 20% 的测试集上进行了评估，以确保在各种环境中的鲁棒性和泛化能力。结果显示，尽管该模型当前由于数据源和算力的限制还无法给出完整精确的道路网络，在结果中各主干道路的预测表明使用先进建模技术预测道路网络扩展具有显著潜力。这种方式无疑有助于增强区域连通性，促进各地区经济增长，并有效地支持战略性的城市规划举措。这项工作大大推动了道路网络预测技术的发展，并强调了其在通过增强道路网络建设来改善经济和社会基础设施方面的重要性。
  ],
  authors: (
    name: "钟熠",
  ),
  index-terms: ("道路网络预测", "深度学习", "城市建设", "Focal Loss", "U-Net"),
  bibliography: bibliography("refs.bib"),
)


= 引入

随着道路基础设施建设的需求不断增长，迫切需要一个能够适应不断变化的地理环境和社会情况的道路网络扩展预测框架。传统的道路规划方法主要依赖于耗时的人工调查和人工的综合数据分析，但在现在快速的城市化和复杂地理特征相互作用的背景下，这些方法越来越显得不足。因此，将地理空间数据和计算机技术结合起来变得尤为重要。近年来，遥感技术的发展和在线地图的广泛应用，使得我们可以获得丰富的地理空间数据，包括来自政府、组织甚至个人提供的高程数据以及道路和水系数据等等。这些数据集为利用深度学习技术创新改变城市规划方法提供了巨大机遇。

当前，越来越多复杂的机器学习模型的发明，加上日渐增强的算力（如GPU等）以及多样化的训练数据集的激增，为深度学习在多个领域的革命性应用奠定了基础。在地理空间分析领域，深度学习已在解决诸如交通网络识别@AI-road-network-prediction、交通情况预测@deep-learning-road-traffic-prediction 等复杂的预测任务中发挥了关键作用。这些进展有力地证明了深度学习技术在地理空间预测中的可行性，从而使得通过机器学习技术增强道路规划成为一种可能。

在这样的大背景下，本研究采用了以图像分割任务著称的U-Net架构@U-Net，并进一步通过AdamW优化器和Focal Loss @Focal-Loss 优化，用于预测道路网络扩展的可能性。模型通过处理地理空间数据（包括从shapefile格式的详细河流、道路网络和住宅区位置，以及以TIFF格式储存的高程数据，从而为预测道路网络的扩展提供了全面及足量的数据基础。为了解决数据中类别不平衡这一关键问题，本研究使用了带有位置权重张量的Focal Loss，以解决数据集中道路矢量因为像素比例过小而模型很难学习的问题。

该模型的评估使用了60%训练集、20%验证集和20%测试集的数据划分。这样的分配确保了验证和测试框架的稳定，以确保模型在不同地理环境中都能够准确和可泛化地进行预测。本研究的最终期望是利用先进的深度学习技术来提高道路网络预测的准确性和效率，和展示如何有效利用这些技术进步来支持经济增长和战略性的城市规划。

本文结构如下：引言之后的第二部分回顾了地理空间数据分析和道路网络预测领域的相关研究。第三部分介绍了方法论，包括数据准备、模型配置和训练过程。第四部分展示了研究结果并讨论了发现的意义。第五部分对结果进行了解释并讨论了当前的局限性。最后，第六部分对论文进行了总结，提出了对该领域的贡献以及未来研究的潜在方向。

= 相关研究

深度学习的兴起显著提升了利用地理空间数据预测和分析道路网络及交通流量的能力。本节回顾了该领域的最新进展，重点介绍了与本研究相关的方法、应用及其成果。

== 深度学习

#figure(
  image("CNN.drawio.svg", width: 80%), 
  caption: [神经网络的一种简单示例]
)

=== 卷积神经网络 (CNN)

卷积神经网络（CNN）构成了人工智能图像分析的基石，尤其是在处理类似网格状数据结构（如图像）的应用中尤为重要。卷积神经网络最早由LeCun等人于1998年提出 @CNN，该网络由专门用于特征提取和分类的层构成——包括卷积层、池化层和全连接层。这种架构使其非常适合用于图像和视频识别任务。

=== U-Net 架构

在卷积神经网络的基础上，Ronneberger、Fischer和Brox于2015年提出的U-Net架构 @U-Net 是一个专门用于图像分割的深度学习架构。该架构的“U形”设计包含一个收缩路径（用于捕捉信息），以及一个对称的扩展路径（以实现精确的定位）。虽然这个架构在创始之初主要用于医学领域的图像分割和识别，它在地理空间分析任务中表现也尤为出色，例如道路网络预测，因为在本项目中，从复杂背景中准确地分割出细节特征的信息（如道路）并精确地确定其位置非常重要。

#figure(
  image("u-net-illustration-correct-scale2.svg", width: 80%),
  caption: [标准 U-Net 架构 (本图来自 @U-Net)]
)

=== AdamW 优化器

深度学习架构（如U-Net）的优化通过AdamW优化器得到了极大增强。AdamW由Loshchilov和Hutter于2017年开发 @AdamW，通过将权重衰减与优化步骤解耦，改进了传统的Adam优化器。这一调整解决了原始Adam算法中的收敛问题，提升了深度学习模型训练中的泛化能力。在道路预测的神经网络训练中，AdamW尤为有用，因为在处理多样化数据时，模型的鲁棒性至关重要。

=== Focal Loss

为了在不平衡数据集上有效训练神经网络，Lin等人于2017年引入了 Focal Loss @Focal-Loss。这种高级损失函数通过根据分类的确定性调整标准的交叉熵损失，增强了模型对难以分类样本的关注。在道路网络预测的背景下，因道路在地理区域中的占比显著偏低，Focal Loss 在提高模型检测能力方面起到了关键作用，确保道路矢量这种微小但重要的特征不会被忽视。

== 相关应用

=== 多模态数据集在道路预测中的应用

在《AI驱动的多模态数据道路网络预测》一文中，研究人员探讨了各种数据融合方法在道路网络检测中的有效性 @AI-road-network-prediction。研究人员结合了卫星影像与GPS轨迹数据，并应用了不同的融合技术，如早期融合和后期融合。结果表明，尤其是通过拼接的早期融合在交并比（IoU）指标上优于后期融合，强调了集成方法在提高道路网络预测方面的潜力。该方法与我们的研究方法高度相关，因为它验证了多模态数据集成在提高预测准确性方面的优势。

=== 用于交通预测的深度学习框架

在《基于深度学习的道路交通预测框架》中，提出了一种结合卷积神经网络（CNN）和循环神经网络（RNN）的新型深度学习模型，用于提高交通预测的准确性 @deep-learning-road-traffic-prediction。该框架通过 CNN 处理空间数据，通过 RNN 处理时间动态，专门针对城市交通模式。这种双重方法与我们模型的架构相一致，其中空间和时间特征对于准确的道路预测至关重要。

=== 用于预测的地理空间数据的转化

《地理空间数据与图像的转换：用于交通预测的深度学习框架》一文中探讨了将地理空间数据转换为类图像格式，以便利用深度学习算法的过程 @geospatial-data-to-images。该研究解决了将传统地理空间数据预处理为适应神经网络架构的必要步骤，这与我们研究中采用的预处理步骤直接相关。

=== 大数据方法在交通流量预测中的应用

最后，《基于大数据的交通流量预测：深度学习方法》展示了深度学习如何有效处理大规模数据集以预测交通状况 @traffic-flow-prediction。通过建立能够处理大量数据的深度神经网络架构，该研究为扩展和管理大规模数据输入提供了宝贵的见解，这是我们工作中的关键方面，因为数据量对预测性能有显著影响。

这些研究共同展示了深度学习技术在处理和预测道路和交通数据集中复杂模式方面的强大能力。我们的研究基于这些基础，旨在进一步完善和扩展这些技术的应用，预测现有的道路模式和潜在的扩展方式，以增强区域间的连接性和促进经济增长。

= 数据处理和研究方法

== 数据处理

考虑到我们通过一个先进的深度学习模型来解决这个问题，数据集的选择和应用就变得极其重要。在这个部分，我将会介绍在这个项目中应用的几个数据获取途径和处理数据的方法。 

=== 数据来源

本项目的数据主要来自多个机构：国家基础地理信息中心、全国地理信息资源目录服务、国家地球系统科学数据中心和地理空间数据云。在设计道路时，我们应考虑的因素包括高度、坡度、土壤、气候和居民区等。此外，评估道路设计应重点关注降低成本和促进区域间的连接性。然而，由于数据和算力的限制，我们仅选择高度（DEM，`tiff`）、河流（`shp`）和居民区（`shp`）作为输入示例，将道路网络作为输出。

=== 裁切和重投影

为了提高效率并降低计算成本，我们将数据裁剪为 $2000 times 2000$ 像素。由于数据来自不同来源，我们可以使用 `geopandas` 读取并重新投影河流、居民区、道路和边界数据为 EPSG:4490 @epsg4490 坐标系。

=== 数据可视化

我们选择宁夏回族自治区作为示例。以下图表是根据前面提到的地理空间数据生成的图片。我们使用了 `matplotlib` 来完成这一步。

图中蓝色的线条代表河流，红色的线条表示道路，而这些灰色的点是居民区。

#figure(
  image("GPD-Ningxia.png", width: 80%),
  caption: [宁夏回族自治区的地理空间数据]
)

在重新投影坐标系并将所有单个高度文件合并后，我们可以直接使用 `Rasterio` 和 `Matplotlib` 可视化高度数据。正如之前提到的，为了节省资源，我们将高度数据也裁剪为 $2000 times 2000$ 像素。

#figure(
  image("clipped_DEM.png", width: 80%),
  caption: [宁夏回族自治区的高程数据（裁剪后）]
)

=== 重新编排数据结构

为了将这些数据作为输入源一同使用，我们可以将上述所有数据合并起来，该数据为 $2000 times 2000$ 像素。这样，输入源将包含 3 个通道，输出将是单通道。

接下来，数据需要分为三个数据加载器：60% 用于训练集，20% 用于验证集，剩下的用于测试集。

== 架构

U-Net 架构最初是为医学用途开发的，特别适合处理图像的分割任务。通过将其与其他深度学习技术结合，我们可以利用这一强大的框架来有效预测道路网络。U-Net 因其出色的图像分割能力和高效的结构，非常适合这一任务。

=== 模型结构

我们采用 U-Net 架构，这是一种最初为生物医学图像分割设计的卷积神经网络。该架构在需要精确定位的情况下非常有效，因为它具有两个特殊结构：收缩部分用于捕捉上下文信息，扩展部分用于实现精确定位。通过实现多个由两组卷积、批归一化和ReLU激活函数组成的双重卷积模块，U-Net 架构在道路分割中的表现尤为突出，因为它能够捕捉到来自卫星图像和高程图等地理空间数据中不同大小的复杂特征，并将它们精确定位到输出上。

该架构包含多个下采样层，允许特征数量增加的同时减少数据的空间尺寸，然后通过跳跃机制将对应的下采样层特征与上采样层的特征交织，应用一个或多个上采样层来恢复特征的尺寸。这种机制保留了在下采样过程中通常会丢失的细节信息，从而确保模型在训练和预测过程中能恢复道路的关键信息。

#figure(
    kind: "algorithm",
    supplement: [伪代码],
    caption: [U-Net 架构],
    pseudocode-list(booktabs: true, stroke: none)[
        - *Define* U-Net Architecture:
          - Input Channels = 3
          - Output Channels = 1 (Binary mask)
          - *Layers*:
            + DoubleConv ($text("Input") -> 64 text("channels")$)
            + DownSampling ($64 -> 128$, $128 -> 256$, $256 -> 512$, $512 -> 512$)
            + UpSampling with skip connections ($512 -> 256 -> 128 -> 64$)
            + Output Convolution ($64 -> 1$, Sigmoid Activation)
    ]
)

=== 损失函数

#figure(
    kind: "algorithm",
    supplement: [伪代码],
    caption: [损失函数],
    pseudocode-list(booktabs: true, stroke: none)[
        - *Define* FocalLoss:
          - *Parameters*: $alpha=0.5$, $gamma=3$
          - *Functionality*:
            - Compute BCE Loss with logits
            - Adjust loss focus based on ground truth relevance and prediction probability
        
        - *Define* ConnectivityLoss:
          - *Parameters*: $alpha=0.001$
          - *Functionality*:
            - Use a $3 times 3$ convolution kernel to dilate predicted masks
            - Compute the mean difference between dilated mask and original prediction
    ]
)

为了解决类别不平衡和道路网络预测的难题，我们的模型采用了两种特殊的损失函数：Focal Loss 和自定义的 Connectivity Loss。Focal Loss 旨在调整模型的重点，使其更关注难以分类的实例（因为道路的像素在整体标签中占比极少）。在这种情况下，通过引入调节因子，该损失函数改进了模型对少数类的表现，将普通的交叉熵损失转变为一种对难分类样本更为敏感的损失函数。它降低了对已经分类良好的实例的损失权重，使模型更加集中于难处理或预测错误的样本，即覆盖地理区域很小的这些道路数据。在本研究中，我们使用 $text(alpha)$-平衡的 Focal Loss 变体：

$ "FL"(p_t) = -alpha_t (1-p_t)^gamma log(p_t). $

Connectivity Loss 进一步增强了模型的预测能力，该损失可以使得预测出的道路网络具有空间连贯性。它对未预测出连续道路的情况施加惩罚，从而提高模型预测出互联道路的可能性。因此，这种损失函数组合的最终输出是一个更具连贯性的道路地图，这让该模型预测的结果在实际的城市设计和交通优化中更加可用。

#figure(
    kind: "algorithm",
    supplement: [伪代码], 
    caption: [连通性损失],
    pseudocode-list(booktabs: true, stroke: none)[
      - *Function* connectivity_loss(mask, alpha=0.001):
        - *Description*:
          - Calculate loss to encourage spatial connectivity in predicted road masks.
        - *Steps*:
          + Create a 3x3 kernel filled with ones.
          + Apply dilation using convolution with padding to maintain dimensions.
             + dilated = conv2d(mask, kernel, padding=1)
          + Convert dilated output to binary (values > 0 set to 1).
          + Calculate the difference between the dilated mask and original mask.
          + Compute the mean of the difference and scale by alpha.
             + $"CL" = "mean"("dilated" - "mask") times alpha$
        - *Return*:
          - connectivity_loss: Scalar value representing the loss due to disconnected areas.
    ]
)

=== 训练过程

我们通过多轮训练优化 U-Net 模型，在训练时使用了 AdamW 优化器，这是一种包含权重衰减的 Adam 优化器变体，用于防止过拟合。模型的训练数据集被分为 60% 的训练集、20% 的验证集和 20% 的测试集，确保对不同地理空间特征的全面覆盖和泛化能力。

每次训练会将输入数据批量通过网络进行前向传播，然后利用前面提到的 Focal Loss 和 Connectivity Loss 损失函数计算损失值。通过反向传播调整模型权重，以最小化这些损失。AdamW 优化器通过结合权重衰减和动量估计来调整学习率，帮助模型高效收敛。间歇性的验证运行为模型性能提供反馈，防止过拟合，并确保模型对新数据的泛化能力。该训练方案不仅确保模型能够学会预测道路，还确保了其预测结果在道路网络分析和规划中能够具有实际的应用价值。

#figure(
    kind: "algorithm",
    supplement: [伪代码],
    caption: [训练过程],
    pseudocode-list(booktabs: true, stroke: none)[
      + *For* each epoch:
        + Perform forward pass with U-Net on training data
        + Calculate Focal Loss and Connectivity Loss
        + Backpropagate total loss
        + Update model weights using AdamW optimizer
        + Validate model performance on the validation set
        + Record training and validation losses
    ]
)

= Results

= Discussion

= Conclusion
