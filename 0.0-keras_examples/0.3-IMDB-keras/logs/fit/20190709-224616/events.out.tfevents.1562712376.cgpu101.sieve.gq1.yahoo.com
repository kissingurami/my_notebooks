       ЃK"	   NFIзAbrain.Event:2Oл$ўH     .юђ~	RєNFIзA"ё
r
dense_1_inputPlaceholder*
dtype0*(
_output_shapes
:џџџџџџџџџш*
shape:џџџџџџџџџш
m
dense_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"ш     
_
dense_1/random_uniform/minConst*
valueB
 *
Н*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *
=*
dtype0*
_output_shapes
: 
Љ
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
T0*
dtype0* 
_output_shapes
:
ш*
seed2ф~*
seedБџх)
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0* 
_output_shapes
:
ш

dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0* 
_output_shapes
:
ш

dense_1/kernel
VariableV2*
dtype0* 
_output_shapes
:
ш*
	container *
shape:
ш*
shared_name 
О
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
validate_shape(* 
_output_shapes
:
ш*
use_locking(*
T0*!
_class
loc:@dense_1/kernel
}
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
ш
\
dense_1/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
z
dense_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Њ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:
r
dense_1/bias/readIdentitydense_1/bias*
_output_shapes	
:*
T0*
_class
loc:@dense_1/bias

dense_1/MatMulMatMuldense_1_inputdense_1/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
X
dense_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
f
$dropout_1/keras_learning_phase/inputConst*
dtype0
*
_output_shapes
: *
value	B
 Z 

dropout_1/keras_learning_phasePlaceholderWithDefault$dropout_1/keras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 

dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
T0
*
_output_shapes
: 
c
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
: 
s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 

dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*(
_output_shapes
:џџџџџџџџџ
Б
dropout_1/cond/mul/SwitchSwitchdense_1/Reludropout_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
z
dropout_1/cond/dropout/rateConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:
{
dropout_1/cond/dropout/sub/xConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?
}
dropout_1/cond/dropout/subSubdropout_1/cond/dropout/sub/xdropout_1/cond/dropout/rate*
T0*
_output_shapes
: 

)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
С
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
T0*
dtype0*(
_output_shapes
:џџџџџџџџџ*
seed2Мл*
seedБџх)
Ї
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
У
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:џџџџџџџџџ
Е
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_1/cond/dropout/addAdddropout_1/cond/dropout/sub%dropout_1/cond/dropout/random_uniform*
T0*(
_output_shapes
:џџџџџџџџџ
t
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_1/cond/dropout/truedivRealDivdropout_1/cond/muldropout_1/cond/dropout/sub*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/truedivdropout_1/cond/dropout/Floor*
T0*(
_output_shapes
:џџџџџџџџџ
Џ
dropout_1/cond/Switch_1Switchdense_1/Reludropout_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
m
dense_2/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *PEнН*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *PEн=
Љ
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
dtype0*
_output_shapes
:	*
seed2зЯЏ*
seedБџх)*
T0
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 

dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes
:	

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes
:	*
T0

dense_2/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes
:	*
	container *
shape:	
Н
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*!
_class
loc:@dense_2/kernel
|
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	
Z
dense_2/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Љ
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_2/bias
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

dense_2/MatMulMatMuldropout_1/cond/Mergedense_2/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
]
dense_2/SoftmaxSoftmaxdense_2/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
]
RMSprop/lr/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *o:
n

RMSprop/lr
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Њ
RMSprop/lr/AssignAssign
RMSprop/lrRMSprop/lr/initial_value*
use_locking(*
T0*
_class
loc:@RMSprop/lr*
validate_shape(*
_output_shapes
: 
g
RMSprop/lr/readIdentity
RMSprop/lr*
T0*
_class
loc:@RMSprop/lr*
_output_shapes
: 
^
RMSprop/rho/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
RMSprop/rho
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ў
RMSprop/rho/AssignAssignRMSprop/rhoRMSprop/rho/initial_value*
T0*
_class
loc:@RMSprop/rho*
validate_shape(*
_output_shapes
: *
use_locking(
j
RMSprop/rho/readIdentityRMSprop/rho*
_output_shapes
: *
T0*
_class
loc:@RMSprop/rho
`
RMSprop/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
RMSprop/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ж
RMSprop/decay/AssignAssignRMSprop/decayRMSprop/decay/initial_value*
use_locking(*
T0* 
_class
loc:@RMSprop/decay*
validate_shape(*
_output_shapes
: 
p
RMSprop/decay/readIdentityRMSprop/decay*
T0* 
_class
loc:@RMSprop/decay*
_output_shapes
: 
b
 RMSprop/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
v
RMSprop/iterations
VariableV2*
shape: *
shared_name *
dtype0	*
_output_shapes
: *
	container 
Ъ
RMSprop/iterations/AssignAssignRMSprop/iterations RMSprop/iterations/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*%
_class
loc:@RMSprop/iterations

RMSprop/iterations/readIdentityRMSprop/iterations*
T0	*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 

dense_2_targetPlaceholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
q
dense_2_sample_weightsPlaceholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
r
'loss/dense_2_loss/Sum/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ѕ
loss/dense_2_loss/SumSumdense_2/Softmax'loss/dense_2_loss/Sum/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
~
loss/dense_2_loss/truedivRealDivdense_2/Softmaxloss/dense_2_loss/Sum*'
_output_shapes
:џџџџџџџџџ*
T0
\
loss/dense_2_loss/ConstConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
\
loss/dense_2_loss/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
o
loss/dense_2_loss/subSubloss/dense_2_loss/sub/xloss/dense_2_loss/Const*
T0*
_output_shapes
: 

'loss/dense_2_loss/clip_by_value/MinimumMinimumloss/dense_2_loss/truedivloss/dense_2_loss/sub*
T0*'
_output_shapes
:џџџџџџџџџ

loss/dense_2_loss/clip_by_valueMaximum'loss/dense_2_loss/clip_by_value/Minimumloss/dense_2_loss/Const*'
_output_shapes
:џџџџџџџџџ*
T0
o
loss/dense_2_loss/LogLogloss/dense_2_loss/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
u
loss/dense_2_loss/mulMuldense_2_targetloss/dense_2_loss/Log*
T0*'
_output_shapes
:џџџџџџџџџ
t
)loss/dense_2_loss/Sum_1/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ћ
loss/dense_2_loss/Sum_1Sumloss/dense_2_loss/mul)loss/dense_2_loss/Sum_1/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0
c
loss/dense_2_loss/NegNegloss/dense_2_loss/Sum_1*#
_output_shapes
:џџџџџџџџџ*
T0
k
(loss/dense_2_loss/Mean/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
Њ
loss/dense_2_loss/MeanMeanloss/dense_2_loss/Neg(loss/dense_2_loss/Mean/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0
|
loss/dense_2_loss/mul_1Mulloss/dense_2_loss/Meandense_2_sample_weights*#
_output_shapes
:џџџџџџџџџ*
T0
a
loss/dense_2_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/dense_2_loss/NotEqualNotEqualdense_2_sample_weightsloss/dense_2_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ

loss/dense_2_loss/CastCastloss/dense_2_loss/NotEqual*

SrcT0
*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0
c
loss/dense_2_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_2_loss/Mean_1Meanloss/dense_2_loss/Castloss/dense_2_loss/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

loss/dense_2_loss/truediv_1RealDivloss/dense_2_loss/mul_1loss/dense_2_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ*
T0
c
loss/dense_2_loss/Const_2Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_2_loss/Mean_2Meanloss/dense_2_loss/truediv_1loss/dense_2_loss/Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
O

loss/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
V
loss/mulMul
loss/mul/xloss/dense_2_loss/Mean_2*
T0*
_output_shapes
: 
g
metrics/acc/ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ

metrics/acc/ArgMaxArgMaxdense_2_targetmetrics/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
i
metrics/acc/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ

metrics/acc/ArgMax_1ArgMaxdense_2/Softmaxmetrics/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*#
_output_shapes
:џџџџџџџџџ*
T0	
x
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

 training/RMSprop/gradients/ShapeConst*
valueB *
_class
loc:@loss/mul*
dtype0*
_output_shapes
: 

$training/RMSprop/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?*
_class
loc:@loss/mul
П
training/RMSprop/gradients/FillFill training/RMSprop/gradients/Shape$training/RMSprop/gradients/grad_ys_0*
T0*

index_type0*
_class
loc:@loss/mul*
_output_shapes
: 
Ќ
,training/RMSprop/gradients/loss/mul_grad/MulMultraining/RMSprop/gradients/Fillloss/dense_2_loss/Mean_2*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
 
.training/RMSprop/gradients/loss/mul_grad/Mul_1Multraining/RMSprop/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
Н
Ftraining/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:*+
_class!
loc:@loss/dense_2_loss/Mean_2
Ѓ
@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/ReshapeReshape.training/RMSprop/gradients/loss/mul_grad/Mul_1Ftraining/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Reshape/shape*
T0*
Tshape0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
_output_shapes
:
Ц
>training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/ShapeShapeloss/dense_2_loss/truediv_1*
T0*
out_type0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
_output_shapes
:
Д
=training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/TileTile@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Reshape>training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Shape*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0
Ш
@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Shape_1Shapeloss/dense_2_loss/truediv_1*
T0*
out_type0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
_output_shapes
:
А
@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB *+
_class!
loc:@loss/dense_2_loss/Mean_2
Е
>training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/ConstConst*
valueB: *+
_class!
loc:@loss/dense_2_loss/Mean_2*
dtype0*
_output_shapes
:
В
=training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/ProdProd@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Shape_1>training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Const*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
_output_shapes
: 
З
@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Const_1Const*
valueB: *+
_class!
loc:@loss/dense_2_loss/Mean_2*
dtype0*
_output_shapes
:
Ж
?training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Prod_1Prod@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Shape_2@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Const_1*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
_output_shapes
: 
Б
Btraining/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Maximum/yConst*
value	B :*+
_class!
loc:@loss/dense_2_loss/Mean_2*
dtype0*
_output_shapes
: 

@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/MaximumMaximum?training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Prod_1Btraining/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
_output_shapes
: 

Atraining/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/floordivFloorDiv=training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Prod@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Maximum*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
_output_shapes
: 
ѕ
=training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/CastCastAtraining/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/floordiv*

SrcT0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
Truncate( *
_output_shapes
: *

DstT0
Є
@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/truedivRealDiv=training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Tile=training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2
Ш
Atraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/ShapeShapeloss/dense_2_loss/mul_1*
T0*
out_type0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*
_output_shapes
:
Ж
Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Shape_1Const*
valueB *.
_class$
" loc:@loss/dense_2_loss/truediv_1*
dtype0*
_output_shapes
: 
п
Qtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/ShapeCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Shape_1*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/RealDivRealDiv@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/truedivloss/dense_2_loss/Mean_1*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
Ю
?training/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/SumSumCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/RealDivQtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/BroadcastGradientArgs*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
О
Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/ReshapeReshape?training/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/SumAtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Shape*
T0*
Tshape0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
Н
?training/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/NegNegloss/dense_2_loss/mul_1*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ

Etraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/RealDiv_1RealDiv?training/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Negloss/dense_2_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1

Etraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/RealDiv_2RealDivEtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/RealDiv_1loss/dense_2_loss/Mean_1*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
­
?training/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/mulMul@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/truedivEtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/RealDiv_2*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
Ю
Atraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Sum_1Sum?training/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/mulStraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*
_output_shapes
:
З
Etraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Reshape_1ReshapeAtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Sum_1Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Shape_1*
T0*
Tshape0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*
_output_shapes
: 
П
=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/ShapeShapeloss/dense_2_loss/Mean*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/dense_2_loss/mul_1
С
?training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Shape_1Shapedense_2_sample_weights*
T0*
out_type0**
_class 
loc:@loss/dense_2_loss/mul_1*
_output_shapes
:
Я
Mtraining/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Shape?training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Shape_1*
T0**
_class 
loc:@loss/dense_2_loss/mul_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
;training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/MulMulCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Reshapedense_2_sample_weights*#
_output_shapes
:џџџџџџџџџ*
T0**
_class 
loc:@loss/dense_2_loss/mul_1
К
;training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/SumSum;training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/MulMtraining/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/BroadcastGradientArgs*
T0**
_class 
loc:@loss/dense_2_loss/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ў
?training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/ReshapeReshape;training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Sum=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0**
_class 
loc:@loss/dense_2_loss/mul_1
ћ
=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Mul_1Mulloss/dense_2_loss/MeanCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Reshape*
T0**
_class 
loc:@loss/dense_2_loss/mul_1*#
_output_shapes
:џџџџџџџџџ
Р
=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Sum_1Sum=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Mul_1Otraining/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/dense_2_loss/mul_1*
_output_shapes
:
Д
Atraining/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Reshape_1Reshape=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Sum_1?training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Shape_1*
T0*
Tshape0**
_class 
loc:@loss/dense_2_loss/mul_1*#
_output_shapes
:џџџџџџџџџ
М
<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/ShapeShapeloss/dense_2_loss/Neg*
T0*
out_type0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
:
Ј
;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*)
_class
loc:@loss/dense_2_loss/Mean
і
:training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/addAdd(loss/dense_2_loss/Mean/reduction_indices;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Size*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_2_loss/Mean

:training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/modFloorMod:training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/add;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Size*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_2_loss/Mean
Г
>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape_1Const*
valueB: *)
_class
loc:@loss/dense_2_loss/Mean*
dtype0*
_output_shapes
:
Џ
Btraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/range/startConst*
value	B : *)
_class
loc:@loss/dense_2_loss/Mean*
dtype0*
_output_shapes
: 
Џ
Btraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/range/deltaConst*
value	B :*)
_class
loc:@loss/dense_2_loss/Mean*
dtype0*
_output_shapes
: 
н
<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/rangeRangeBtraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/range/start;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/SizeBtraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/range/delta*
_output_shapes
:*

Tidx0*)
_class
loc:@loss/dense_2_loss/Mean
Ў
Atraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Fill/valueConst*
value	B :*)
_class
loc:@loss/dense_2_loss/Mean*
dtype0*
_output_shapes
: 
І
;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/FillFill>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape_1Atraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Fill/value*
T0*

index_type0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
: 
Ѓ
Dtraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/DynamicStitchDynamicStitch<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/range:training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/mod<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Fill*
N*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_2_loss/Mean
­
@training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*)
_class
loc:@loss/dense_2_loss/Mean
Ё
>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/MaximumMaximumDtraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/DynamicStitch@training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
:

?training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/floordivFloorDiv<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
:
З
>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/ReshapeReshape?training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/ReshapeDtraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/DynamicStitch*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0*)
_class
loc:@loss/dense_2_loss/Mean
Џ
;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/TileTile>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Reshape?training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/floordiv*

Tmultiples0*
T0*)
_class
loc:@loss/dense_2_loss/Mean*#
_output_shapes
:џџџџџџџџџ
О
>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape_2Shapeloss/dense_2_loss/Neg*
T0*
out_type0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
:
П
>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape_3Shapeloss/dense_2_loss/Mean*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/dense_2_loss/Mean
Б
<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/ConstConst*
valueB: *)
_class
loc:@loss/dense_2_loss/Mean*
dtype0*
_output_shapes
:
Њ
;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/ProdProd>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape_2<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
: 
Г
>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Const_1Const*
valueB: *)
_class
loc:@loss/dense_2_loss/Mean*
dtype0*
_output_shapes
:
Ў
=training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Prod_1Prod>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape_3>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/dense_2_loss/Mean
Џ
Btraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*)
_class
loc:@loss/dense_2_loss/Mean

@training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum_1Maximum=training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Prod_1Btraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum_1/y*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
: 

Atraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/floordiv_1FloorDiv;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Prod@training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum_1*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
: 
ё
;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/CastCastAtraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/floordiv_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0*)
_class
loc:@loss/dense_2_loss/Mean

>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/truedivRealDiv;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Tile;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0*)
_class
loc:@loss/dense_2_loss/Mean
и
9training/RMSprop/gradients/loss/dense_2_loss/Neg_grad/NegNeg>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/truediv*
T0*(
_class
loc:@loss/dense_2_loss/Neg*#
_output_shapes
:џџџџџџџџџ
О
=training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/ShapeShapeloss/dense_2_loss/mul*
T0*
out_type0**
_class 
loc:@loss/dense_2_loss/Sum_1*
_output_shapes
:
Њ
<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/SizeConst*
value	B :**
_class 
loc:@loss/dense_2_loss/Sum_1*
dtype0*
_output_shapes
: 
ј
;training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/addAdd)loss/dense_2_loss/Sum_1/reduction_indices<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Size*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1*
_output_shapes
: 

;training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/modFloorMod;training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/add<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Size*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1*
_output_shapes
: 
Ў
?training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB **
_class 
loc:@loss/dense_2_loss/Sum_1
Б
Ctraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/range/startConst*
value	B : **
_class 
loc:@loss/dense_2_loss/Sum_1*
dtype0*
_output_shapes
: 
Б
Ctraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/range/deltaConst*
value	B :**
_class 
loc:@loss/dense_2_loss/Sum_1*
dtype0*
_output_shapes
: 
т
=training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/rangeRangeCtraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/range/start<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/SizeCtraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/range/delta**
_class 
loc:@loss/dense_2_loss/Sum_1*
_output_shapes
:*

Tidx0
А
Btraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Fill/valueConst*
value	B :**
_class 
loc:@loss/dense_2_loss/Sum_1*
dtype0*
_output_shapes
: 
Ј
<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/FillFill?training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Shape_1Btraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Fill/value*
T0*

index_type0**
_class 
loc:@loss/dense_2_loss/Sum_1*
_output_shapes
: 
Љ
Etraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/DynamicStitchDynamicStitch=training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/range;training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/mod=training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Shape<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Fill*
N*
_output_shapes
:*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1
Џ
Atraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Maximum/yConst*
value	B :**
_class 
loc:@loss/dense_2_loss/Sum_1*
dtype0*
_output_shapes
: 
Ѕ
?training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/MaximumMaximumEtraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/DynamicStitchAtraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Maximum/y*
_output_shapes
:*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1

@training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/floordivFloorDiv=training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Shape?training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Maximum*
_output_shapes
:*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1
С
?training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/ReshapeReshape9training/RMSprop/gradients/loss/dense_2_loss/Neg_grad/NegEtraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/DynamicStitch*
T0*
Tshape0**
_class 
loc:@loss/dense_2_loss/Sum_1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
З
<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/TileTile?training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Reshape@training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/floordiv*

Tmultiples0*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1*'
_output_shapes
:џџџџџџџџџ
Г
;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/ShapeShapedense_2_target*
T0*
out_type0*(
_class
loc:@loss/dense_2_loss/mul*
_output_shapes
:
М
=training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Shape_1Shapeloss/dense_2_loss/Log*
T0*
out_type0*(
_class
loc:@loss/dense_2_loss/mul*
_output_shapes
:
Ч
Ktraining/RMSprop/gradients/loss/dense_2_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Shape=training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_2_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ё
9training/RMSprop/gradients/loss/dense_2_loss/mul_grad/MulMul<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Tileloss/dense_2_loss/Log*
T0*(
_class
loc:@loss/dense_2_loss/mul*'
_output_shapes
:џџџџџџџџџ
В
9training/RMSprop/gradients/loss/dense_2_loss/mul_grad/SumSum9training/RMSprop/gradients/loss/dense_2_loss/mul_grad/MulKtraining/RMSprop/gradients/loss/dense_2_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/dense_2_loss/mul
Г
=training/RMSprop/gradients/loss/dense_2_loss/mul_grad/ReshapeReshape9training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Sum;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_2_loss/mul*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ь
;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Mul_1Muldense_2_target<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Tile*'
_output_shapes
:џџџџџџџџџ*
T0*(
_class
loc:@loss/dense_2_loss/mul
И
;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Sum_1Sum;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Mul_1Mtraining/RMSprop/gradients/loss/dense_2_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/dense_2_loss/mul
А
?training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Reshape_1Reshape;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Sum_1=training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0*(
_class
loc:@loss/dense_2_loss/mul

@training/RMSprop/gradients/loss/dense_2_loss/Log_grad/Reciprocal
Reciprocalloss/dense_2_loss/clip_by_value@^training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Reshape_1*
T0*(
_class
loc:@loss/dense_2_loss/Log*'
_output_shapes
:џџџџџџџџџ

9training/RMSprop/gradients/loss/dense_2_loss/Log_grad/mulMul?training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Reshape_1@training/RMSprop/gradients/loss/dense_2_loss/Log_grad/Reciprocal*
T0*(
_class
loc:@loss/dense_2_loss/Log*'
_output_shapes
:џџџџџџџџџ
р
Etraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/ShapeShape'loss/dense_2_loss/clip_by_value/Minimum*
T0*
out_type0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
_output_shapes
:
О
Gtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Shape_1Const*
valueB *2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
dtype0*
_output_shapes
: 
є
Gtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Shape_2Shape9training/RMSprop/gradients/loss/dense_2_loss/Log_grad/mul*
T0*
out_type0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
_output_shapes
:
Ф
Ktraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *2
_class(
&$loc:@loss/dense_2_loss/clip_by_value
л
Etraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/zerosFillGtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Shape_2Ktraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/zeros/Const*
T0*

index_type0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ

Ltraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/GreaterEqualGreaterEqual'loss/dense_2_loss/clip_by_value/Minimumloss/dense_2_loss/Const*'
_output_shapes
:џџџџџџџџџ*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value
я
Utraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsEtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/ShapeGtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Shape_1*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ftraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/SelectSelectLtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/GreaterEqual9training/RMSprop/gradients/loss/dense_2_loss/Log_grad/mulEtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/zeros*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ

Htraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Select_1SelectLtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/GreaterEqualEtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/zeros9training/RMSprop/gradients/loss/dense_2_loss/Log_grad/mul*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ
н
Ctraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/SumSumFtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/SelectUtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/BroadcastGradientArgs*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
_output_shapes
:*

Tidx0*
	keep_dims( 
в
Gtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/ReshapeReshapeCtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/SumEtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Shape*
T0*
Tshape0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ
у
Etraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Sum_1SumHtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Select_1Wtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
_output_shapes
:
Ч
Itraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Reshape_1ReshapeEtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Sum_1Gtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Shape_1*
T0*
Tshape0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
_output_shapes
: 
т
Mtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/ShapeShapeloss/dense_2_loss/truediv*
_output_shapes
:*
T0*
out_type0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum
Ю
Otraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Shape_1Const*
valueB *:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*
dtype0*
_output_shapes
: 

Otraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Shape_2ShapeGtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Reshape*
_output_shapes
:*
T0*
out_type0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum
д
Straining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*
dtype0*
_output_shapes
: 
ћ
Mtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/zerosFillOtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Shape_2Straining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
ў
Qtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/dense_2_loss/truedivloss/dense_2_loss/sub*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum

]training/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/ShapeOtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Shape_1*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Б
Ntraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/SelectSelectQtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/LessEqualGtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/ReshapeMtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/zeros*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
Г
Ptraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Select_1SelectQtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/LessEqualMtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/zerosGtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum
§
Ktraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/SumSumNtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Select]training/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*
_output_shapes
:*

Tidx0*
	keep_dims( 
ђ
Otraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/ReshapeReshapeKtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/SumMtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum

Mtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Sum_1SumPtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Select_1_training/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum
ч
Qtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeMtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Sum_1Otraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*
_output_shapes
: 
М
?training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/ShapeShapedense_2/Softmax*
_output_shapes
:*
T0*
out_type0*,
_class"
 loc:@loss/dense_2_loss/truediv
Ф
Atraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Shape_1Shapeloss/dense_2_loss/Sum*
T0*
out_type0*,
_class"
 loc:@loss/dense_2_loss/truediv*
_output_shapes
:
з
Otraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs?training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/ShapeAtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Atraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/RealDivRealDivOtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Reshapeloss/dense_2_loss/Sum*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*'
_output_shapes
:џџџџџџџџџ
Ц
=training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/SumSumAtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/RealDivOtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/BroadcastGradientArgs*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
К
Atraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/ReshapeReshape=training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Sum?training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0*,
_class"
 loc:@loss/dense_2_loss/truediv
Е
=training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/NegNegdense_2/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv

Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/RealDiv_1RealDiv=training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Negloss/dense_2_loss/Sum*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*'
_output_shapes
:џџџџџџџџџ

Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/RealDiv_2RealDivCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/RealDiv_1loss/dense_2_loss/Sum*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*'
_output_shapes
:џџџџџџџџџ
К
=training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/mulMulOtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/ReshapeCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/RealDiv_2*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*'
_output_shapes
:џџџџџџџџџ
Ц
?training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Sum_1Sum=training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/mulQtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/dense_2_loss/truediv
Р
Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Reshape_1Reshape?training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Sum_1Atraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Shape_1*
T0*
Tshape0*,
_class"
 loc:@loss/dense_2_loss/truediv*'
_output_shapes
:џџџџџџџџџ
Д
;training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/ShapeShapedense_2/Softmax*
_output_shapes
:*
T0*
out_type0*(
_class
loc:@loss/dense_2_loss/Sum
І
:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/SizeConst*
value	B :*(
_class
loc:@loss/dense_2_loss/Sum*
dtype0*
_output_shapes
: 
№
9training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/addAdd'loss/dense_2_loss/Sum/reduction_indices:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Size*
T0*(
_class
loc:@loss/dense_2_loss/Sum*
_output_shapes
: 

9training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/modFloorMod9training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/add:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Size*
T0*(
_class
loc:@loss/dense_2_loss/Sum*
_output_shapes
: 
Њ
=training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *(
_class
loc:@loss/dense_2_loss/Sum
­
Atraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/range/startConst*
value	B : *(
_class
loc:@loss/dense_2_loss/Sum*
dtype0*
_output_shapes
: 
­
Atraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/range/deltaConst*
value	B :*(
_class
loc:@loss/dense_2_loss/Sum*
dtype0*
_output_shapes
: 
и
;training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/rangeRangeAtraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/range/start:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/SizeAtraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/range/delta*(
_class
loc:@loss/dense_2_loss/Sum*
_output_shapes
:*

Tidx0
Ќ
@training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Fill/valueConst*
value	B :*(
_class
loc:@loss/dense_2_loss/Sum*
dtype0*
_output_shapes
: 
 
:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/FillFill=training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Shape_1@training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Fill/value*
_output_shapes
: *
T0*

index_type0*(
_class
loc:@loss/dense_2_loss/Sum

Ctraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/DynamicStitchDynamicStitch;training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/range9training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/mod;training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Shape:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Fill*
N*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_2_loss/Sum
Ћ
?training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Maximum/yConst*
value	B :*(
_class
loc:@loss/dense_2_loss/Sum*
dtype0*
_output_shapes
: 

=training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/MaximumMaximumCtraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/DynamicStitch?training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Maximum/y*
T0*(
_class
loc:@loss/dense_2_loss/Sum*
_output_shapes
:

>training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/floordivFloorDiv;training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Shape=training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Maximum*
T0*(
_class
loc:@loss/dense_2_loss/Sum*
_output_shapes
:
Х
=training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/ReshapeReshapeCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Reshape_1Ctraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/DynamicStitch*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0*(
_class
loc:@loss/dense_2_loss/Sum
Џ
:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/TileTile=training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Reshape>training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/floordiv*

Tmultiples0*
T0*(
_class
loc:@loss/dense_2_loss/Sum*'
_output_shapes
:џџџџџџџџџ

training/RMSprop/gradients/AddNAddNAtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Reshape:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Tile*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*
N*'
_output_shapes
:џџџџџџџџџ
Т
3training/RMSprop/gradients/dense_2/Softmax_grad/mulMultraining/RMSprop/gradients/AddNdense_2/Softmax*
T0*"
_class
loc:@dense_2/Softmax*'
_output_shapes
:џџџџџџџџџ
Д
Etraining/RMSprop/gradients/dense_2/Softmax_grad/Sum/reduction_indicesConst*
valueB :
џџџџџџџџџ*"
_class
loc:@dense_2/Softmax*
dtype0*
_output_shapes
: 
Љ
3training/RMSprop/gradients/dense_2/Softmax_grad/SumSum3training/RMSprop/gradients/dense_2/Softmax_grad/mulEtraining/RMSprop/gradients/dense_2/Softmax_grad/Sum/reduction_indices*
T0*"
_class
loc:@dense_2/Softmax*'
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims(
ц
3training/RMSprop/gradients/dense_2/Softmax_grad/subSubtraining/RMSprop/gradients/AddN3training/RMSprop/gradients/dense_2/Softmax_grad/Sum*
T0*"
_class
loc:@dense_2/Softmax*'
_output_shapes
:џџџџџџџџџ
и
5training/RMSprop/gradients/dense_2/Softmax_grad/mul_1Mul3training/RMSprop/gradients/dense_2/Softmax_grad/subdense_2/Softmax*
T0*"
_class
loc:@dense_2/Softmax*'
_output_shapes
:џџџџџџџџџ
с
;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad5training/RMSprop/gradients/dense_2/Softmax_grad/mul_1*
data_formatNHWC*
_output_shapes
:*
T0*"
_class
loc:@dense_2/BiasAdd

5training/RMSprop/gradients/dense_2/MatMul_grad/MatMulMatMul5training/RMSprop/gradients/dense_2/Softmax_grad/mul_1dense_2/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_2/MatMul

7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1MatMuldropout_1/cond/Merge5training/RMSprop/gradients/dense_2/Softmax_grad/mul_1*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul

>training/RMSprop/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch5training/RMSprop/gradients/dense_2/MatMul_grad/MatMuldropout_1/cond/pred_id*
T0*!
_class
loc:@dense_2/MatMul*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
Й
!training/RMSprop/gradients/SwitchSwitchdense_1/Reludropout_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
Ј
#training/RMSprop/gradients/IdentityIdentity#training/RMSprop/gradients/Switch:1*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ
І
"training/RMSprop/gradients/Shape_1Shape#training/RMSprop/gradients/Switch:1*
T0*
out_type0*
_class
loc:@dense_1/Relu*
_output_shapes
:
В
&training/RMSprop/gradients/zeros/ConstConst$^training/RMSprop/gradients/Identity*
valueB
 *    *
_class
loc:@dense_1/Relu*
dtype0*
_output_shapes
: 
к
 training/RMSprop/gradients/zerosFill"training/RMSprop/gradients/Shape_1&training/RMSprop/gradients/zeros/Const*
T0*

index_type0*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ

Atraining/RMSprop/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge>training/RMSprop/gradients/dropout_1/cond/Merge_grad/cond_grad training/RMSprop/gradients/zeros*
T0*
_class
loc:@dense_1/Relu*
N**
_output_shapes
:џџџџџџџџџ: 
Э
@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/ShapeShapedropout_1/cond/dropout/truediv*
T0*
out_type0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
Э
Btraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/Floor*
_output_shapes
:*
T0*
out_type0*-
_class#
!loc:@dropout_1/cond/dropout/mul
л
Ptraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/ShapeBtraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul

>training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/MulMul@training/RMSprop/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Floor*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
Ц
>training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/SumSum>training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/MulPtraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
П
Btraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape>training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Sum@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@dropout_1/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ

@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Muldropout_1/cond/dropout/truediv@training/RMSprop/gradients/dropout_1/cond/Merge_grad/cond_grad:1*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul
Ь
@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Rtraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
Х
Dtraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Btraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*-
_class#
!loc:@dropout_1/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ
Щ
Dtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/ShapeShapedropout_1/cond/mul*
_output_shapes
:*
T0*
out_type0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
М
Ftraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1Const*
valueB *1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
dtype0*
_output_shapes
: 
ы
Ttraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsDtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/ShapeFtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ftraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/RealDivRealDivBtraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Reshapedropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*(
_output_shapes
:џџџџџџџџџ
к
Btraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/SumSumFtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/RealDivTtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:
Я
Ftraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/ReshapeReshapeBtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/SumDtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
У
Btraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/NegNegdropout_1/cond/mul*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*(
_output_shapes
:џџџџџџџџџ

Htraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1RealDivBtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Negdropout_1/cond/dropout/sub*(
_output_shapes
:џџџџџџџџџ*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv

Htraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2RealDivHtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1dropout_1/cond/dropout/sub*(
_output_shapes
:џџџџџџџџџ*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
Н
Btraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/mulMulBtraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/ReshapeHtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2*(
_output_shapes
:џџџџџџџџџ*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
к
Dtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1SumBtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/mulVtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
У
Htraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Reshape_1ReshapeDtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Ftraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
T0*
Tshape0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
: 
К
8training/RMSprop/gradients/dropout_1/cond/mul_grad/ShapeShapedropout_1/cond/mul/Switch:1*
T0*
out_type0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:
Є
:training/RMSprop/gradients/dropout_1/cond/mul_grad/Shape_1Const*
valueB *%
_class
loc:@dropout_1/cond/mul*
dtype0*
_output_shapes
: 
Л
Htraining/RMSprop/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/RMSprop/gradients/dropout_1/cond/mul_grad/Shape:training/RMSprop/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_1/cond/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ѕ
6training/RMSprop/gradients/dropout_1/cond/mul_grad/MulMulFtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Reshapedropout_1/cond/mul/y*
T0*%
_class
loc:@dropout_1/cond/mul*(
_output_shapes
:џџџџџџџџџ
І
6training/RMSprop/gradients/dropout_1/cond/mul_grad/SumSum6training/RMSprop/gradients/dropout_1/cond/mul_grad/MulHtraining/RMSprop/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*%
_class
loc:@dropout_1/cond/mul

:training/RMSprop/gradients/dropout_1/cond/mul_grad/ReshapeReshape6training/RMSprop/gradients/dropout_1/cond/mul_grad/Sum8training/RMSprop/gradients/dropout_1/cond/mul_grad/Shape*
T0*
Tshape0*%
_class
loc:@dropout_1/cond/mul*(
_output_shapes
:џџџџџџџџџ
ў
8training/RMSprop/gradients/dropout_1/cond/mul_grad/Mul_1Muldropout_1/cond/mul/Switch:1Ftraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Reshape*
T0*%
_class
loc:@dropout_1/cond/mul*(
_output_shapes
:џџџџџџџџџ
Ќ
8training/RMSprop/gradients/dropout_1/cond/mul_grad/Sum_1Sum8training/RMSprop/gradients/dropout_1/cond/mul_grad/Mul_1Jtraining/RMSprop/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 

<training/RMSprop/gradients/dropout_1/cond/mul_grad/Reshape_1Reshape8training/RMSprop/gradients/dropout_1/cond/mul_grad/Sum_1:training/RMSprop/gradients/dropout_1/cond/mul_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*%
_class
loc:@dropout_1/cond/mul
Л
#training/RMSprop/gradients/Switch_1Switchdense_1/Reludropout_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
Њ
%training/RMSprop/gradients/Identity_1Identity#training/RMSprop/gradients/Switch_1*(
_output_shapes
:џџџџџџџџџ*
T0*
_class
loc:@dense_1/Relu
І
"training/RMSprop/gradients/Shape_2Shape#training/RMSprop/gradients/Switch_1*
T0*
out_type0*
_class
loc:@dense_1/Relu*
_output_shapes
:
Ж
(training/RMSprop/gradients/zeros_1/ConstConst&^training/RMSprop/gradients/Identity_1*
valueB
 *    *
_class
loc:@dense_1/Relu*
dtype0*
_output_shapes
: 
о
"training/RMSprop/gradients/zeros_1Fill"training/RMSprop/gradients/Shape_2(training/RMSprop/gradients/zeros_1/Const*
T0*

index_type0*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ

Ctraining/RMSprop/gradients/dropout_1/cond/mul/Switch_grad/cond_gradMerge"training/RMSprop/gradients/zeros_1:training/RMSprop/gradients/dropout_1/cond/mul_grad/Reshape*
T0*
_class
loc:@dense_1/Relu*
N**
_output_shapes
:џџџџџџџџџ: 

!training/RMSprop/gradients/AddN_1AddNAtraining/RMSprop/gradients/dropout_1/cond/Switch_1_grad/cond_gradCtraining/RMSprop/gradients/dropout_1/cond/mul/Switch_grad/cond_grad*
T0*
_class
loc:@dense_1/Relu*
N*(
_output_shapes
:џџџџџџџџџ
Ц
5training/RMSprop/gradients/dense_1/Relu_grad/ReluGradReluGrad!training/RMSprop/gradients/AddN_1dense_1/Relu*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ
т
;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad5training/RMSprop/gradients/dense_1/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes	
:

5training/RMSprop/gradients/dense_1/MatMul_grad/MatMulMatMul5training/RMSprop/gradients/dense_1/Relu_grad/ReluGraddense_1/kernel/read*(
_output_shapes
:џџџџџџџџџш*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul
ћ
7training/RMSprop/gradients/dense_1/MatMul_grad/MatMul_1MatMuldense_1_input5training/RMSprop/gradients/dense_1/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_1/MatMul* 
_output_shapes
:
ш*
transpose_a(*
transpose_b( 
w
&training/RMSprop/zeros/shape_as_tensorConst*
valueB"ш     *
dtype0*
_output_shapes
:
a
training/RMSprop/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ё
training/RMSprop/zerosFill&training/RMSprop/zeros/shape_as_tensortraining/RMSprop/zeros/Const* 
_output_shapes
:
ш*
T0*

index_type0

training/RMSprop/Variable
VariableV2*
shape:
ш*
shared_name *
dtype0* 
_output_shapes
:
ш*
	container 
п
 training/RMSprop/Variable/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/zeros*
validate_shape(* 
_output_shapes
:
ш*
use_locking(*
T0*,
_class"
 loc:@training/RMSprop/Variable

training/RMSprop/Variable/readIdentitytraining/RMSprop/Variable* 
_output_shapes
:
ш*
T0*,
_class"
 loc:@training/RMSprop/Variable
g
training/RMSprop/zeros_1Const*
valueB*    *
dtype0*
_output_shapes	
:

training/RMSprop/Variable_1
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
т
"training/RMSprop/Variable_1/AssignAssigntraining/RMSprop/Variable_1training/RMSprop/zeros_1*
T0*.
_class$
" loc:@training/RMSprop/Variable_1*
validate_shape(*
_output_shapes	
:*
use_locking(

 training/RMSprop/Variable_1/readIdentitytraining/RMSprop/Variable_1*
T0*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes	
:
y
(training/RMSprop/zeros_2/shape_as_tensorConst*
valueB"      *
dtype0*
_output_shapes
:
c
training/RMSprop/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
І
training/RMSprop/zeros_2Fill(training/RMSprop/zeros_2/shape_as_tensortraining/RMSprop/zeros_2/Const*
_output_shapes
:	*
T0*

index_type0

training/RMSprop/Variable_2
VariableV2*
dtype0*
_output_shapes
:	*
	container *
shape:	*
shared_name 
ц
"training/RMSprop/Variable_2/AssignAssigntraining/RMSprop/Variable_2training/RMSprop/zeros_2*
T0*.
_class$
" loc:@training/RMSprop/Variable_2*
validate_shape(*
_output_shapes
:	*
use_locking(
Ѓ
 training/RMSprop/Variable_2/readIdentitytraining/RMSprop/Variable_2*
_output_shapes
:	*
T0*.
_class$
" loc:@training/RMSprop/Variable_2
e
training/RMSprop/zeros_3Const*
dtype0*
_output_shapes
:*
valueB*    

training/RMSprop/Variable_3
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
с
"training/RMSprop/Variable_3/AssignAssigntraining/RMSprop/Variable_3training/RMSprop/zeros_3*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_3*
validate_shape(*
_output_shapes
:

 training/RMSprop/Variable_3/readIdentitytraining/RMSprop/Variable_3*
T0*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
:
b
 training/RMSprop/AssignAdd/valueConst*
dtype0	*
_output_shapes
: *
value	B	 R
И
training/RMSprop/AssignAdd	AssignAddRMSprop/iterations training/RMSprop/AssignAdd/value*
use_locking( *
T0	*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 
x
training/RMSprop/mulMulRMSprop/rho/readtraining/RMSprop/Variable/read* 
_output_shapes
:
ш*
T0
[
training/RMSprop/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/RMSprop/subSubtraining/RMSprop/sub/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/SquareSquare7training/RMSprop/gradients/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
ш*
T0
w
training/RMSprop/mul_1Multraining/RMSprop/subtraining/RMSprop/Square*
T0* 
_output_shapes
:
ш
t
training/RMSprop/addAddtraining/RMSprop/multraining/RMSprop/mul_1* 
_output_shapes
:
ш*
T0
д
training/RMSprop/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/add*
use_locking(*
T0*,
_class"
 loc:@training/RMSprop/Variable*
validate_shape(* 
_output_shapes
:
ш

training/RMSprop/mul_2MulRMSprop/lr/read7training/RMSprop/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
ш
[
training/RMSprop/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training/RMSprop/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  

&training/RMSprop/clip_by_value/MinimumMinimumtraining/RMSprop/addtraining/RMSprop/Const_1*
T0* 
_output_shapes
:
ш

training/RMSprop/clip_by_valueMaximum&training/RMSprop/clip_by_value/Minimumtraining/RMSprop/Const*
T0* 
_output_shapes
:
ш
h
training/RMSprop/SqrtSqrttraining/RMSprop/clip_by_value*
T0* 
_output_shapes
:
ш
]
training/RMSprop/add_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
y
training/RMSprop/add_1Addtraining/RMSprop/Sqrttraining/RMSprop/add_1/y*
T0* 
_output_shapes
:
ш
~
training/RMSprop/truedivRealDivtraining/RMSprop/mul_2training/RMSprop/add_1* 
_output_shapes
:
ш*
T0
w
training/RMSprop/sub_1Subdense_1/kernel/readtraining/RMSprop/truediv*
T0* 
_output_shapes
:
ш
Т
training/RMSprop/Assign_1Assigndense_1/kerneltraining/RMSprop/sub_1*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(* 
_output_shapes
:
ш
w
training/RMSprop/mul_3MulRMSprop/rho/read training/RMSprop/Variable_1/read*
T0*
_output_shapes	
:
]
training/RMSprop/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_2Subtraining/RMSprop/sub_2/xRMSprop/rho/read*
_output_shapes
: *
T0

training/RMSprop/Square_1Square;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
v
training/RMSprop/mul_4Multraining/RMSprop/sub_2training/RMSprop/Square_1*
T0*
_output_shapes	
:
s
training/RMSprop/add_2Addtraining/RMSprop/mul_3training/RMSprop/mul_4*
_output_shapes	
:*
T0
з
training/RMSprop/Assign_2Assigntraining/RMSprop/Variable_1training/RMSprop/add_2*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_1

training/RMSprop/mul_5MulRMSprop/lr/read;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
]
training/RMSprop/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training/RMSprop/Const_3Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_1/MinimumMinimumtraining/RMSprop/add_2training/RMSprop/Const_3*
T0*
_output_shapes	
:

 training/RMSprop/clip_by_value_1Maximum(training/RMSprop/clip_by_value_1/Minimumtraining/RMSprop/Const_2*
_output_shapes	
:*
T0
g
training/RMSprop/Sqrt_1Sqrt training/RMSprop/clip_by_value_1*
T0*
_output_shapes	
:
]
training/RMSprop/add_3/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
v
training/RMSprop/add_3Addtraining/RMSprop/Sqrt_1training/RMSprop/add_3/y*
T0*
_output_shapes	
:
{
training/RMSprop/truediv_1RealDivtraining/RMSprop/mul_5training/RMSprop/add_3*
T0*
_output_shapes	
:
r
training/RMSprop/sub_3Subdense_1/bias/readtraining/RMSprop/truediv_1*
T0*
_output_shapes	
:
Й
training/RMSprop/Assign_3Assigndense_1/biastraining/RMSprop/sub_3*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense_1/bias
{
training/RMSprop/mul_6MulRMSprop/rho/read training/RMSprop/Variable_2/read*
T0*
_output_shapes
:	
]
training/RMSprop/sub_4/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
j
training/RMSprop/sub_4Subtraining/RMSprop/sub_4/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_2Square7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
z
training/RMSprop/mul_7Multraining/RMSprop/sub_4training/RMSprop/Square_2*
T0*
_output_shapes
:	
w
training/RMSprop/add_4Addtraining/RMSprop/mul_6training/RMSprop/mul_7*
_output_shapes
:	*
T0
л
training/RMSprop/Assign_4Assigntraining/RMSprop/Variable_2training/RMSprop/add_4*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_2*
validate_shape(*
_output_shapes
:	

training/RMSprop/mul_8MulRMSprop/lr/read7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
]
training/RMSprop/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training/RMSprop/Const_5Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_2/MinimumMinimumtraining/RMSprop/add_4training/RMSprop/Const_5*
T0*
_output_shapes
:	

 training/RMSprop/clip_by_value_2Maximum(training/RMSprop/clip_by_value_2/Minimumtraining/RMSprop/Const_4*
T0*
_output_shapes
:	
k
training/RMSprop/Sqrt_2Sqrt training/RMSprop/clip_by_value_2*
T0*
_output_shapes
:	
]
training/RMSprop/add_5/yConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
z
training/RMSprop/add_5Addtraining/RMSprop/Sqrt_2training/RMSprop/add_5/y*
T0*
_output_shapes
:	

training/RMSprop/truediv_2RealDivtraining/RMSprop/mul_8training/RMSprop/add_5*
T0*
_output_shapes
:	
x
training/RMSprop/sub_5Subdense_2/kernel/readtraining/RMSprop/truediv_2*
T0*
_output_shapes
:	
С
training/RMSprop/Assign_5Assigndense_2/kerneltraining/RMSprop/sub_5*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	
v
training/RMSprop/mul_9MulRMSprop/rho/read training/RMSprop/Variable_3/read*
_output_shapes
:*
T0
]
training/RMSprop/sub_6/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_6Subtraining/RMSprop/sub_6/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_3Square;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
v
training/RMSprop/mul_10Multraining/RMSprop/sub_6training/RMSprop/Square_3*
T0*
_output_shapes
:
s
training/RMSprop/add_6Addtraining/RMSprop/mul_9training/RMSprop/mul_10*
T0*
_output_shapes
:
ж
training/RMSprop/Assign_6Assigntraining/RMSprop/Variable_3training/RMSprop/add_6*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_3

training/RMSprop/mul_11MulRMSprop/lr/read;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
]
training/RMSprop/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training/RMSprop/Const_7Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_3/MinimumMinimumtraining/RMSprop/add_6training/RMSprop/Const_7*
T0*
_output_shapes
:

 training/RMSprop/clip_by_value_3Maximum(training/RMSprop/clip_by_value_3/Minimumtraining/RMSprop/Const_6*
T0*
_output_shapes
:
f
training/RMSprop/Sqrt_3Sqrt training/RMSprop/clip_by_value_3*
T0*
_output_shapes
:
]
training/RMSprop/add_7/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
u
training/RMSprop/add_7Addtraining/RMSprop/Sqrt_3training/RMSprop/add_7/y*
_output_shapes
:*
T0
{
training/RMSprop/truediv_3RealDivtraining/RMSprop/mul_11training/RMSprop/add_7*
T0*
_output_shapes
:
q
training/RMSprop/sub_7Subdense_2/bias/readtraining/RMSprop/truediv_3*
T0*
_output_shapes
:
И
training/RMSprop/Assign_7Assigndense_2/biastraining/RMSprop/sub_7*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
Д
training/group_depsNoOp	^loss/mul^metrics/acc/Mean^training/RMSprop/Assign^training/RMSprop/AssignAdd^training/RMSprop/Assign_1^training/RMSprop/Assign_2^training/RMSprop/Assign_3^training/RMSprop/Assign_4^training/RMSprop/Assign_5^training/RMSprop/Assign_6^training/RMSprop/Assign_7
0

group_depsNoOp	^loss/mul^metrics/acc/Mean

IsVariableInitializedIsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializeddense_2/bias*
dtype0*
_output_shapes
: *
_class
loc:@dense_2/bias

IsVariableInitialized_4IsVariableInitialized
RMSprop/lr*
_class
loc:@RMSprop/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializedRMSprop/rho*
_class
loc:@RMSprop/rho*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializedRMSprop/decay*
dtype0*
_output_shapes
: * 
_class
loc:@RMSprop/decay

IsVariableInitialized_7IsVariableInitializedRMSprop/iterations*%
_class
loc:@RMSprop/iterations*
dtype0	*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedtraining/RMSprop/Variable*,
_class"
 loc:@training/RMSprop/Variable*
dtype0*
_output_shapes
: 
Ђ
IsVariableInitialized_9IsVariableInitializedtraining/RMSprop/Variable_1*.
_class$
" loc:@training/RMSprop/Variable_1*
dtype0*
_output_shapes
: 
Ѓ
IsVariableInitialized_10IsVariableInitializedtraining/RMSprop/Variable_2*.
_class$
" loc:@training/RMSprop/Variable_2*
dtype0*
_output_shapes
: 
Ѓ
IsVariableInitialized_11IsVariableInitializedtraining/RMSprop/Variable_3*.
_class$
" loc:@training/RMSprop/Variable_3*
dtype0*
_output_shapes
: 
ж
initNoOp^RMSprop/decay/Assign^RMSprop/iterations/Assign^RMSprop/lr/Assign^RMSprop/rho/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign!^training/RMSprop/Variable/Assign#^training/RMSprop/Variable_1/Assign#^training/RMSprop/Variable_2/Assign#^training/RMSprop/Variable_3/Assign"w(Oссg     1эђ	ќъNFIзAJдЯ
Ё  
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	

C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.13.12
b'unknown'ё
r
dense_1_inputPlaceholder*
dtype0*(
_output_shapes
:џџџџџџџџџш*
shape:џџџџџџџџџш
m
dense_1/random_uniform/shapeConst*
valueB"ш     *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *
Н
_
dense_1/random_uniform/maxConst*
valueB
 *
=*
dtype0*
_output_shapes
: 
Љ
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed2ф~* 
_output_shapes
:
ш*
seedБџх)*
T0
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0* 
_output_shapes
:
ш

dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0* 
_output_shapes
:
ш

dense_1/kernel
VariableV2*
shared_name *
dtype0*
	container * 
_output_shapes
:
ш*
shape:
ш
О
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
validate_shape(* 
_output_shapes
:
ш*
use_locking(*
T0*!
_class
loc:@dense_1/kernel
}
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
ш
\
dense_1/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
z
dense_1/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
Њ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:

dense_1/MatMulMatMuldense_1_inputdense_1/kernel/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:џџџџџџџџџ

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
X
dense_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
f
$dropout_1/keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 

dropout_1/keras_learning_phasePlaceholderWithDefault$dropout_1/keras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 

dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
T0
*
_output_shapes
: 
c
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
: *
T0

s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ?

dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*(
_output_shapes
:џџџџџџџџџ
Б
dropout_1/cond/mul/SwitchSwitchdense_1/Reludropout_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
z
dropout_1/cond/dropout/rateConst^dropout_1/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
_output_shapes
:*
T0*
out_type0
{
dropout_1/cond/dropout/sub/xConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
}
dropout_1/cond/dropout/subSubdropout_1/cond/dropout/sub/xdropout_1/cond/dropout/rate*
T0*
_output_shapes
: 

)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
С
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
T0*
dtype0*
seed2Мл*(
_output_shapes
:џџџџџџџџџ*
seedБџх)
Ї
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
У
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:џџџџџџџџџ
Е
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_1/cond/dropout/addAdddropout_1/cond/dropout/sub%dropout_1/cond/dropout/random_uniform*
T0*(
_output_shapes
:џџџџџџџџџ
t
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*(
_output_shapes
:џџџџџџџџџ*
T0

dropout_1/cond/dropout/truedivRealDivdropout_1/cond/muldropout_1/cond/dropout/sub*
T0*(
_output_shapes
:џџџџџџџџџ

dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/truedivdropout_1/cond/dropout/Floor*(
_output_shapes
:џџџџџџџџџ*
T0
Џ
dropout_1/cond/Switch_1Switchdense_1/Reludropout_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
m
dense_2/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *PEнН*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *PEн=*
dtype0*
_output_shapes
: 
Љ
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0*
seed2зЯЏ*
_output_shapes
:	*
seedБџх)
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 

dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes
:	

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes
:	

dense_2/kernel
VariableV2*
shape:	*
shared_name *
dtype0*
	container *
_output_shapes
:	
Н
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*!
_class
loc:@dense_2/kernel
|
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	
Z
dense_2/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_2/bias
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
Љ
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_2/bias
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

dense_2/MatMulMatMuldropout_1/cond/Mergedense_2/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
]
dense_2/SoftmaxSoftmaxdense_2/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
]
RMSprop/lr/initial_valueConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
n

RMSprop/lr
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
Њ
RMSprop/lr/AssignAssign
RMSprop/lrRMSprop/lr/initial_value*
T0*
_class
loc:@RMSprop/lr*
validate_shape(*
_output_shapes
: *
use_locking(
g
RMSprop/lr/readIdentity
RMSprop/lr*
T0*
_class
loc:@RMSprop/lr*
_output_shapes
: 
^
RMSprop/rho/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
o
RMSprop/rho
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
Ў
RMSprop/rho/AssignAssignRMSprop/rhoRMSprop/rho/initial_value*
use_locking(*
T0*
_class
loc:@RMSprop/rho*
validate_shape(*
_output_shapes
: 
j
RMSprop/rho/readIdentityRMSprop/rho*
T0*
_class
loc:@RMSprop/rho*
_output_shapes
: 
`
RMSprop/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
RMSprop/decay
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
Ж
RMSprop/decay/AssignAssignRMSprop/decayRMSprop/decay/initial_value*
use_locking(*
T0* 
_class
loc:@RMSprop/decay*
validate_shape(*
_output_shapes
: 
p
RMSprop/decay/readIdentityRMSprop/decay*
T0* 
_class
loc:@RMSprop/decay*
_output_shapes
: 
b
 RMSprop/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
v
RMSprop/iterations
VariableV2*
shared_name *
dtype0	*
	container *
_output_shapes
: *
shape: 
Ъ
RMSprop/iterations/AssignAssignRMSprop/iterations RMSprop/iterations/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*%
_class
loc:@RMSprop/iterations

RMSprop/iterations/readIdentityRMSprop/iterations*
_output_shapes
: *
T0	*%
_class
loc:@RMSprop/iterations

dense_2_targetPlaceholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
q
dense_2_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
r
'loss/dense_2_loss/Sum/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ѕ
loss/dense_2_loss/SumSumdense_2/Softmax'loss/dense_2_loss/Sum/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims(
~
loss/dense_2_loss/truedivRealDivdense_2/Softmaxloss/dense_2_loss/Sum*'
_output_shapes
:џџџџџџџџџ*
T0
\
loss/dense_2_loss/ConstConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
\
loss/dense_2_loss/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
o
loss/dense_2_loss/subSubloss/dense_2_loss/sub/xloss/dense_2_loss/Const*
T0*
_output_shapes
: 

'loss/dense_2_loss/clip_by_value/MinimumMinimumloss/dense_2_loss/truedivloss/dense_2_loss/sub*
T0*'
_output_shapes
:џџџџџџџџџ

loss/dense_2_loss/clip_by_valueMaximum'loss/dense_2_loss/clip_by_value/Minimumloss/dense_2_loss/Const*
T0*'
_output_shapes
:џџџџџџџџџ
o
loss/dense_2_loss/LogLogloss/dense_2_loss/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
u
loss/dense_2_loss/mulMuldense_2_targetloss/dense_2_loss/Log*'
_output_shapes
:џџџџџџџџџ*
T0
t
)loss/dense_2_loss/Sum_1/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ћ
loss/dense_2_loss/Sum_1Sumloss/dense_2_loss/mul)loss/dense_2_loss/Sum_1/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 
c
loss/dense_2_loss/NegNegloss/dense_2_loss/Sum_1*
T0*#
_output_shapes
:џџџџџџџџџ
k
(loss/dense_2_loss/Mean/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
Њ
loss/dense_2_loss/MeanMeanloss/dense_2_loss/Neg(loss/dense_2_loss/Mean/reduction_indices*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( *
T0
|
loss/dense_2_loss/mul_1Mulloss/dense_2_loss/Meandense_2_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
a
loss/dense_2_loss/NotEqual/yConst*
dtype0*
_output_shapes
: *
valueB
 *    

loss/dense_2_loss/NotEqualNotEqualdense_2_sample_weightsloss/dense_2_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ

loss/dense_2_loss/CastCastloss/dense_2_loss/NotEqual*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:џџџџџџџџџ
c
loss/dense_2_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_2_loss/Mean_1Meanloss/dense_2_loss/Castloss/dense_2_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0

loss/dense_2_loss/truediv_1RealDivloss/dense_2_loss/mul_1loss/dense_2_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ*
T0
c
loss/dense_2_loss/Const_2Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_2_loss/Mean_2Meanloss/dense_2_loss/truediv_1loss/dense_2_loss/Const_2*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/dense_2_loss/Mean_2*
_output_shapes
: *
T0
g
metrics/acc/ArgMax/dimensionConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

metrics/acc/ArgMaxArgMaxdense_2_targetmetrics/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
i
metrics/acc/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ

metrics/acc/ArgMax_1ArgMaxdense_2/Softmaxmetrics/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
x
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:џџџџџџџџџ
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0

 training/RMSprop/gradients/ShapeConst*
_class
loc:@loss/mul*
valueB *
dtype0*
_output_shapes
: 

$training/RMSprop/gradients/grad_ys_0Const*
_class
loc:@loss/mul*
valueB
 *  ?*
dtype0*
_output_shapes
: 
П
training/RMSprop/gradients/FillFill training/RMSprop/gradients/Shape$training/RMSprop/gradients/grad_ys_0*
T0*
_class
loc:@loss/mul*

index_type0*
_output_shapes
: 
Ќ
,training/RMSprop/gradients/loss/mul_grad/MulMultraining/RMSprop/gradients/Fillloss/dense_2_loss/Mean_2*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
 
.training/RMSprop/gradients/loss/mul_grad/Mul_1Multraining/RMSprop/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
Н
Ftraining/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*+
_class!
loc:@loss/dense_2_loss/Mean_2*
valueB:
Ѓ
@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/ReshapeReshape.training/RMSprop/gradients/loss/mul_grad/Mul_1Ftraining/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Reshape/shape*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
Tshape0*
_output_shapes
:
Ц
>training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/ShapeShapeloss/dense_2_loss/truediv_1*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
out_type0
Д
=training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/TileTile@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Reshape>training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Shape*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2
Ш
@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Shape_1Shapeloss/dense_2_loss/truediv_1*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
out_type0*
_output_shapes
:
А
@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Shape_2Const*+
_class!
loc:@loss/dense_2_loss/Mean_2*
valueB *
dtype0*
_output_shapes
: 
Е
>training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/ConstConst*+
_class!
loc:@loss/dense_2_loss/Mean_2*
valueB: *
dtype0*
_output_shapes
:
В
=training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/ProdProd@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Shape_1>training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Const*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
_output_shapes
: *
	keep_dims( *

Tidx0
З
@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Const_1Const*+
_class!
loc:@loss/dense_2_loss/Mean_2*
valueB: *
dtype0*
_output_shapes
:
Ж
?training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Prod_1Prod@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Shape_2@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2
Б
Btraining/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Maximum/yConst*+
_class!
loc:@loss/dense_2_loss/Mean_2*
value	B :*
dtype0*
_output_shapes
: 

@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/MaximumMaximum?training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Prod_1Btraining/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
_output_shapes
: 

Atraining/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/floordivFloorDiv=training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Prod@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Maximum*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2*
_output_shapes
: 
ѕ
=training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/CastCastAtraining/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/floordiv*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0*+
_class!
loc:@loss/dense_2_loss/Mean_2
Є
@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/truedivRealDiv=training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Tile=training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0*+
_class!
loc:@loss/dense_2_loss/Mean_2
Ш
Atraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/ShapeShapeloss/dense_2_loss/mul_1*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*
out_type0*
_output_shapes
:
Ж
Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Shape_1Const*.
_class$
" loc:@loss/dense_2_loss/truediv_1*
valueB *
dtype0*
_output_shapes
: 
п
Qtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/ShapeCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Shape_1*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/RealDivRealDiv@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/truedivloss/dense_2_loss/Mean_1*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
Ю
?training/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/SumSumCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/RealDivQtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1
О
Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/ReshapeReshape?training/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/SumAtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*
Tshape0
Н
?training/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/NegNegloss/dense_2_loss/mul_1*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ

Etraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/RealDiv_1RealDiv?training/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Negloss/dense_2_loss/Mean_1*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ

Etraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/RealDiv_2RealDivEtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/RealDiv_1loss/dense_2_loss/Mean_1*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
­
?training/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/mulMul@training/RMSprop/gradients/loss/dense_2_loss/Mean_2_grad/truedivEtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/RealDiv_2*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*#
_output_shapes
:џџџџџџџџџ
Ю
Atraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Sum_1Sum?training/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/mulStraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*
_output_shapes
:
З
Etraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Reshape_1ReshapeAtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Sum_1Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Shape_1*
T0*.
_class$
" loc:@loss/dense_2_loss/truediv_1*
Tshape0*
_output_shapes
: 
П
=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/ShapeShapeloss/dense_2_loss/Mean*
T0**
_class 
loc:@loss/dense_2_loss/mul_1*
out_type0*
_output_shapes
:
С
?training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Shape_1Shapedense_2_sample_weights*
T0**
_class 
loc:@loss/dense_2_loss/mul_1*
out_type0*
_output_shapes
:
Я
Mtraining/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Shape?training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Shape_1*
T0**
_class 
loc:@loss/dense_2_loss/mul_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
;training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/MulMulCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Reshapedense_2_sample_weights*
T0**
_class 
loc:@loss/dense_2_loss/mul_1*#
_output_shapes
:џџџџџџџџџ
К
;training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/SumSum;training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/MulMtraining/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/BroadcastGradientArgs*
T0**
_class 
loc:@loss/dense_2_loss/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
Ў
?training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/ReshapeReshape;training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Sum=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Shape*
T0**
_class 
loc:@loss/dense_2_loss/mul_1*
Tshape0*#
_output_shapes
:џџџџџџџџџ
ћ
=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Mul_1Mulloss/dense_2_loss/MeanCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_1_grad/Reshape*#
_output_shapes
:џџџџџџџџџ*
T0**
_class 
loc:@loss/dense_2_loss/mul_1
Р
=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Sum_1Sum=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Mul_1Otraining/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/BroadcastGradientArgs:1*
T0**
_class 
loc:@loss/dense_2_loss/mul_1*
_output_shapes
:*
	keep_dims( *

Tidx0
Д
Atraining/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Reshape_1Reshape=training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Sum_1?training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/Shape_1*
T0**
_class 
loc:@loss/dense_2_loss/mul_1*
Tshape0*#
_output_shapes
:џџџџџџџџџ
М
<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/ShapeShapeloss/dense_2_loss/Neg*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
out_type0*
_output_shapes
:
Ј
;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/SizeConst*)
_class
loc:@loss/dense_2_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
і
:training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/addAdd(loss/dense_2_loss/Mean/reduction_indices;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
: 

:training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/modFloorMod:training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/add;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Size*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_2_loss/Mean
Г
>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*)
_class
loc:@loss/dense_2_loss/Mean*
valueB: 
Џ
Btraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/range/startConst*)
_class
loc:@loss/dense_2_loss/Mean*
value	B : *
dtype0*
_output_shapes
: 
Џ
Btraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/range/deltaConst*)
_class
loc:@loss/dense_2_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
н
<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/rangeRangeBtraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/range/start;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/SizeBtraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/range/delta*

Tidx0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
:
Ў
Atraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *)
_class
loc:@loss/dense_2_loss/Mean*
value	B :
І
;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/FillFill>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape_1Atraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Fill/value*
T0*)
_class
loc:@loss/dense_2_loss/Mean*

index_type0*
_output_shapes
: 
Ѓ
Dtraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/DynamicStitchDynamicStitch<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/range:training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/mod<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Fill*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
N*
_output_shapes
:
­
@training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum/yConst*)
_class
loc:@loss/dense_2_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
Ё
>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/MaximumMaximumDtraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/DynamicStitch@training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
:

?training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/floordivFloorDiv<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
:
З
>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/ReshapeReshape?training/RMSprop/gradients/loss/dense_2_loss/mul_1_grad/ReshapeDtraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/DynamicStitch*#
_output_shapes
:џџџџџџџџџ*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
Tshape0
Џ
;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/TileTile>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Reshape?training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/floordiv*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0*)
_class
loc:@loss/dense_2_loss/Mean
О
>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape_2Shapeloss/dense_2_loss/Neg*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
out_type0
П
>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape_3Shapeloss/dense_2_loss/Mean*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
out_type0
Б
<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/ConstConst*)
_class
loc:@loss/dense_2_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Њ
;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/ProdProd>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape_2<training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
: 
Г
>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Const_1Const*)
_class
loc:@loss/dense_2_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Ў
=training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Prod_1Prod>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Shape_3>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Const_1*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0
Џ
Btraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum_1/yConst*)
_class
loc:@loss/dense_2_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 

@training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum_1Maximum=training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Prod_1Btraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum_1/y*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
: 

Atraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/floordiv_1FloorDiv;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Prod@training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Maximum_1*
T0*)
_class
loc:@loss/dense_2_loss/Mean*
_output_shapes
: 
ё
;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/CastCastAtraining/RMSprop/gradients/loss/dense_2_loss/Mean_grad/floordiv_1*

SrcT0*)
_class
loc:@loss/dense_2_loss/Mean*
Truncate( *

DstT0*
_output_shapes
: 

>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/truedivRealDiv;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Tile;training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/dense_2_loss/Mean*#
_output_shapes
:џџџџџџџџџ
и
9training/RMSprop/gradients/loss/dense_2_loss/Neg_grad/NegNeg>training/RMSprop/gradients/loss/dense_2_loss/Mean_grad/truediv*
T0*(
_class
loc:@loss/dense_2_loss/Neg*#
_output_shapes
:џџџџџџџџџ
О
=training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/ShapeShapeloss/dense_2_loss/mul*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1*
out_type0*
_output_shapes
:
Њ
<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/SizeConst**
_class 
loc:@loss/dense_2_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 
ј
;training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/addAdd)loss/dense_2_loss/Sum_1/reduction_indices<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Size*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1*
_output_shapes
: 

;training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/modFloorMod;training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/add<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Size*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1*
_output_shapes
: 
Ў
?training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Shape_1Const**
_class 
loc:@loss/dense_2_loss/Sum_1*
valueB *
dtype0*
_output_shapes
: 
Б
Ctraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/range/startConst*
dtype0*
_output_shapes
: **
_class 
loc:@loss/dense_2_loss/Sum_1*
value	B : 
Б
Ctraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/range/deltaConst*
dtype0*
_output_shapes
: **
_class 
loc:@loss/dense_2_loss/Sum_1*
value	B :
т
=training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/rangeRangeCtraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/range/start<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/SizeCtraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/range/delta*
_output_shapes
:*

Tidx0**
_class 
loc:@loss/dense_2_loss/Sum_1
А
Btraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Fill/valueConst**
_class 
loc:@loss/dense_2_loss/Sum_1*
value	B :*
dtype0*
_output_shapes
: 
Ј
<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/FillFill?training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Shape_1Btraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Fill/value*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1*

index_type0*
_output_shapes
: 
Љ
Etraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/DynamicStitchDynamicStitch=training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/range;training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/mod=training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Shape<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Fill*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1*
N*
_output_shapes
:
Џ
Atraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: **
_class 
loc:@loss/dense_2_loss/Sum_1*
value	B :
Ѕ
?training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/MaximumMaximumEtraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/DynamicStitchAtraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Maximum/y*
_output_shapes
:*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1

@training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/floordivFloorDiv=training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Shape?training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Maximum*
_output_shapes
:*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1
С
?training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/ReshapeReshape9training/RMSprop/gradients/loss/dense_2_loss/Neg_grad/NegEtraining/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/DynamicStitch*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
З
<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/TileTile?training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Reshape@training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/floordiv*

Tmultiples0*
T0**
_class 
loc:@loss/dense_2_loss/Sum_1*'
_output_shapes
:џџџџџџџџџ
Г
;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/ShapeShapedense_2_target*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_2_loss/mul*
out_type0
М
=training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Shape_1Shapeloss/dense_2_loss/Log*
T0*(
_class
loc:@loss/dense_2_loss/mul*
out_type0*
_output_shapes
:
Ч
Ktraining/RMSprop/gradients/loss/dense_2_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Shape=training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_2_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ё
9training/RMSprop/gradients/loss/dense_2_loss/mul_grad/MulMul<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Tileloss/dense_2_loss/Log*
T0*(
_class
loc:@loss/dense_2_loss/mul*'
_output_shapes
:џџџџџџџџџ
В
9training/RMSprop/gradients/loss/dense_2_loss/mul_grad/SumSum9training/RMSprop/gradients/loss/dense_2_loss/mul_grad/MulKtraining/RMSprop/gradients/loss/dense_2_loss/mul_grad/BroadcastGradientArgs*
T0*(
_class
loc:@loss/dense_2_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
Г
=training/RMSprop/gradients/loss/dense_2_loss/mul_grad/ReshapeReshape9training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Sum;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Shape*
T0*(
_class
loc:@loss/dense_2_loss/mul*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ь
;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Mul_1Muldense_2_target<training/RMSprop/gradients/loss/dense_2_loss/Sum_1_grad/Tile*'
_output_shapes
:џџџџџџџџџ*
T0*(
_class
loc:@loss/dense_2_loss/mul
И
;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Sum_1Sum;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Mul_1Mtraining/RMSprop/gradients/loss/dense_2_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/dense_2_loss/mul
А
?training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Reshape_1Reshape;training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Sum_1=training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*(
_class
loc:@loss/dense_2_loss/mul*
Tshape0

@training/RMSprop/gradients/loss/dense_2_loss/Log_grad/Reciprocal
Reciprocalloss/dense_2_loss/clip_by_value@^training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0*(
_class
loc:@loss/dense_2_loss/Log

9training/RMSprop/gradients/loss/dense_2_loss/Log_grad/mulMul?training/RMSprop/gradients/loss/dense_2_loss/mul_grad/Reshape_1@training/RMSprop/gradients/loss/dense_2_loss/Log_grad/Reciprocal*'
_output_shapes
:џџџџџџџџџ*
T0*(
_class
loc:@loss/dense_2_loss/Log
р
Etraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/ShapeShape'loss/dense_2_loss/clip_by_value/Minimum*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
out_type0*
_output_shapes
:
О
Gtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Shape_1Const*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
valueB *
dtype0*
_output_shapes
: 
є
Gtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Shape_2Shape9training/RMSprop/gradients/loss/dense_2_loss/Log_grad/mul*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
out_type0*
_output_shapes
:
Ф
Ktraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/zeros/ConstConst*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
valueB
 *    *
dtype0*
_output_shapes
: 
л
Etraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/zerosFillGtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Shape_2Ktraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/zeros/Const*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*

index_type0*'
_output_shapes
:џџџџџџџџџ

Ltraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/GreaterEqualGreaterEqual'loss/dense_2_loss/clip_by_value/Minimumloss/dense_2_loss/Const*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ
я
Utraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsEtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/ShapeGtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Shape_1*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ftraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/SelectSelectLtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/GreaterEqual9training/RMSprop/gradients/loss/dense_2_loss/Log_grad/mulEtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/zeros*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ

Htraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Select_1SelectLtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/GreaterEqualEtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/zeros9training/RMSprop/gradients/loss/dense_2_loss/Log_grad/mul*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*'
_output_shapes
:џџџџџџџџџ
н
Ctraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/SumSumFtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/SelectUtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/BroadcastGradientArgs*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0
в
Gtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/ReshapeReshapeCtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/SumEtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Shape*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
Tshape0*'
_output_shapes
:џџџџџџџџџ
у
Etraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Sum_1SumHtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Select_1Wtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/BroadcastGradientArgs:1*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0
Ч
Itraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Reshape_1ReshapeEtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Sum_1Gtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Shape_1*
T0*2
_class(
&$loc:@loss/dense_2_loss/clip_by_value*
Tshape0*
_output_shapes
: 
т
Mtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/ShapeShapeloss/dense_2_loss/truediv*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:
Ю
Otraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Shape_1Const*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*
valueB *
dtype0*
_output_shapes
: 

Otraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Shape_2ShapeGtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Reshape*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:
д
Straining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/zeros/ConstConst*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*
valueB
 *    *
dtype0*
_output_shapes
: 
ћ
Mtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/zerosFillOtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Shape_2Straining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*

index_type0*'
_output_shapes
:џџџџџџџџџ
ў
Qtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/dense_2_loss/truedivloss/dense_2_loss/sub*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ

]training/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/ShapeOtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Shape_1*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Б
Ntraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/SelectSelectQtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/LessEqualGtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/ReshapeMtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/zeros*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
Г
Ptraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Select_1SelectQtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/LessEqualMtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/zerosGtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value_grad/Reshape*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
§
Ktraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/SumSumNtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Select]training/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*
_output_shapes
:
ђ
Otraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/ReshapeReshapeKtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/SumMtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Shape*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*
Tshape0*'
_output_shapes
:џџџџџџџџџ

Mtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Sum_1SumPtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Select_1_training/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*
_output_shapes
:*
	keep_dims( *

Tidx0
ч
Qtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeMtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Sum_1Otraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Shape_1*
T0*:
_class0
.,loc:@loss/dense_2_loss/clip_by_value/Minimum*
Tshape0*
_output_shapes
: 
М
?training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/ShapeShapedense_2/Softmax*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*
out_type0*
_output_shapes
:
Ф
Atraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Shape_1Shapeloss/dense_2_loss/Sum*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*
out_type0*
_output_shapes
:
з
Otraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs?training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/ShapeAtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Atraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/RealDivRealDivOtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/Reshapeloss/dense_2_loss/Sum*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*'
_output_shapes
:џџџџџџџџџ
Ц
=training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/SumSumAtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/RealDivOtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/BroadcastGradientArgs*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
К
Atraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/ReshapeReshape=training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Sum?training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*
Tshape0
Е
=training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/NegNegdense_2/Softmax*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*'
_output_shapes
:џџџџџџџџџ

Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/RealDiv_1RealDiv=training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Negloss/dense_2_loss/Sum*'
_output_shapes
:џџџџџџџџџ*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv

Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/RealDiv_2RealDivCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/RealDiv_1loss/dense_2_loss/Sum*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*'
_output_shapes
:џџџџџџџџџ
К
=training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/mulMulOtraining/RMSprop/gradients/loss/dense_2_loss/clip_by_value/Minimum_grad/ReshapeCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/RealDiv_2*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*'
_output_shapes
:џџџџџџџџџ
Ц
?training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Sum_1Sum=training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/mulQtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv
Р
Ctraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Reshape_1Reshape?training/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Sum_1Atraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*
Tshape0
Д
;training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/ShapeShapedense_2/Softmax*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_2_loss/Sum*
out_type0
І
:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/SizeConst*(
_class
loc:@loss/dense_2_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
№
9training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/addAdd'loss/dense_2_loss/Sum/reduction_indices:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Size*
_output_shapes
: *
T0*(
_class
loc:@loss/dense_2_loss/Sum

9training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/modFloorMod9training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/add:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Size*
T0*(
_class
loc:@loss/dense_2_loss/Sum*
_output_shapes
: 
Њ
=training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Shape_1Const*
dtype0*
_output_shapes
: *(
_class
loc:@loss/dense_2_loss/Sum*
valueB 
­
Atraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/range/startConst*(
_class
loc:@loss/dense_2_loss/Sum*
value	B : *
dtype0*
_output_shapes
: 
­
Atraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/range/deltaConst*(
_class
loc:@loss/dense_2_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
и
;training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/rangeRangeAtraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/range/start:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/SizeAtraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/range/delta*
_output_shapes
:*

Tidx0*(
_class
loc:@loss/dense_2_loss/Sum
Ќ
@training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Fill/valueConst*(
_class
loc:@loss/dense_2_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 
 
:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/FillFill=training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Shape_1@training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Fill/value*
_output_shapes
: *
T0*(
_class
loc:@loss/dense_2_loss/Sum*

index_type0

Ctraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/DynamicStitchDynamicStitch;training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/range9training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/mod;training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Shape:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Fill*
T0*(
_class
loc:@loss/dense_2_loss/Sum*
N*
_output_shapes
:
Ћ
?training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Maximum/yConst*(
_class
loc:@loss/dense_2_loss/Sum*
value	B :*
dtype0*
_output_shapes
: 

=training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/MaximumMaximumCtraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/DynamicStitch?training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Maximum/y*
T0*(
_class
loc:@loss/dense_2_loss/Sum*
_output_shapes
:

>training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/floordivFloorDiv;training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Shape=training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Maximum*
T0*(
_class
loc:@loss/dense_2_loss/Sum*
_output_shapes
:
Х
=training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/ReshapeReshapeCtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Reshape_1Ctraining/RMSprop/gradients/loss/dense_2_loss/Sum_grad/DynamicStitch*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*(
_class
loc:@loss/dense_2_loss/Sum*
Tshape0
Џ
:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/TileTile=training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Reshape>training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/floordiv*

Tmultiples0*
T0*(
_class
loc:@loss/dense_2_loss/Sum*'
_output_shapes
:џџџџџџџџџ

training/RMSprop/gradients/AddNAddNAtraining/RMSprop/gradients/loss/dense_2_loss/truediv_grad/Reshape:training/RMSprop/gradients/loss/dense_2_loss/Sum_grad/Tile*
T0*,
_class"
 loc:@loss/dense_2_loss/truediv*
N*'
_output_shapes
:џџџџџџџџџ
Т
3training/RMSprop/gradients/dense_2/Softmax_grad/mulMultraining/RMSprop/gradients/AddNdense_2/Softmax*
T0*"
_class
loc:@dense_2/Softmax*'
_output_shapes
:џџџџџџџџџ
Д
Etraining/RMSprop/gradients/dense_2/Softmax_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *"
_class
loc:@dense_2/Softmax*
valueB :
џџџџџџџџџ
Љ
3training/RMSprop/gradients/dense_2/Softmax_grad/SumSum3training/RMSprop/gradients/dense_2/Softmax_grad/mulEtraining/RMSprop/gradients/dense_2/Softmax_grad/Sum/reduction_indices*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0*
T0*"
_class
loc:@dense_2/Softmax
ц
3training/RMSprop/gradients/dense_2/Softmax_grad/subSubtraining/RMSprop/gradients/AddN3training/RMSprop/gradients/dense_2/Softmax_grad/Sum*'
_output_shapes
:џџџџџџџџџ*
T0*"
_class
loc:@dense_2/Softmax
и
5training/RMSprop/gradients/dense_2/Softmax_grad/mul_1Mul3training/RMSprop/gradients/dense_2/Softmax_grad/subdense_2/Softmax*
T0*"
_class
loc:@dense_2/Softmax*'
_output_shapes
:џџџџџџџџџ
с
;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad5training/RMSprop/gradients/dense_2/Softmax_grad/mul_1*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC*
_output_shapes
:

5training/RMSprop/gradients/dense_2/MatMul_grad/MatMulMatMul5training/RMSprop/gradients/dense_2/Softmax_grad/mul_1dense_2/kernel/read*
T0*!
_class
loc:@dense_2/MatMul*
transpose_a( *(
_output_shapes
:џџџџџџџџџ*
transpose_b(

7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1MatMuldropout_1/cond/Merge5training/RMSprop/gradients/dense_2/Softmax_grad/mul_1*
T0*!
_class
loc:@dense_2/MatMul*
transpose_a(*
_output_shapes
:	*
transpose_b( 

>training/RMSprop/gradients/dropout_1/cond/Merge_grad/cond_gradSwitch5training/RMSprop/gradients/dense_2/MatMul_grad/MatMuldropout_1/cond/pred_id*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*!
_class
loc:@dense_2/MatMul
Й
!training/RMSprop/gradients/SwitchSwitchdense_1/Reludropout_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
Ј
#training/RMSprop/gradients/IdentityIdentity#training/RMSprop/gradients/Switch:1*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ
І
"training/RMSprop/gradients/Shape_1Shape#training/RMSprop/gradients/Switch:1*
_output_shapes
:*
T0*
_class
loc:@dense_1/Relu*
out_type0
В
&training/RMSprop/gradients/zeros/ConstConst$^training/RMSprop/gradients/Identity*
_class
loc:@dense_1/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
к
 training/RMSprop/gradients/zerosFill"training/RMSprop/gradients/Shape_1&training/RMSprop/gradients/zeros/Const*
T0*
_class
loc:@dense_1/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ

Atraining/RMSprop/gradients/dropout_1/cond/Switch_1_grad/cond_gradMerge>training/RMSprop/gradients/dropout_1/cond/Merge_grad/cond_grad training/RMSprop/gradients/zeros*
T0*
_class
loc:@dense_1/Relu*
N**
_output_shapes
:џџџџџџџџџ: 
Э
@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/ShapeShapedropout_1/cond/dropout/truediv*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0*
_output_shapes
:
Э
Btraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Shape_1Shapedropout_1/cond/dropout/Floor*
_output_shapes
:*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
out_type0
л
Ptraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/ShapeBtraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

>training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/MulMul@training/RMSprop/gradients/dropout_1/cond/Merge_grad/cond_grad:1dropout_1/cond/dropout/Floor*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ
Ц
>training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/SumSum>training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/MulPtraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
П
Btraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/ReshapeReshape>training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Sum@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0

@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Muldropout_1/cond/dropout/truediv@training/RMSprop/gradients/dropout_1/cond/Merge_grad/cond_grad:1*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*(
_output_shapes
:џџџџџџџџџ
Ь
@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Sum@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Mul_1Rtraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
_output_shapes
:
Х
Dtraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape@training/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Sum_1Btraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@dropout_1/cond/dropout/mul*
Tshape0
Щ
Dtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/ShapeShapedropout_1/cond/mul*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
out_type0*
_output_shapes
:
М
Ftraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1Const*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
valueB *
dtype0*
_output_shapes
: 
ы
Ttraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsDtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/ShapeFtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ftraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/RealDivRealDivBtraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/Reshapedropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*(
_output_shapes
:џџџџџџџџџ
к
Btraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/SumSumFtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/RealDivTtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
Я
Ftraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/ReshapeReshapeBtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/SumDtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Shape*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
Tshape0*(
_output_shapes
:џџџџџџџџџ
У
Btraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/NegNegdropout_1/cond/mul*(
_output_shapes
:џџџџџџџџџ*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv

Htraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1RealDivBtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Negdropout_1/cond/dropout/sub*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*(
_output_shapes
:џџџџџџџџџ

Htraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2RealDivHtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_1dropout_1/cond/dropout/sub*(
_output_shapes
:џџџџџџџџџ*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv
Н
Btraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/mulMulBtraining/RMSprop/gradients/dropout_1/cond/dropout/mul_grad/ReshapeHtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/RealDiv_2*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*(
_output_shapes
:џџџџџџџџџ
к
Dtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1SumBtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/mulVtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
У
Htraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Reshape_1ReshapeDtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Sum_1Ftraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@dropout_1/cond/dropout/truediv*
Tshape0*
_output_shapes
: 
К
8training/RMSprop/gradients/dropout_1/cond/mul_grad/ShapeShapedropout_1/cond/mul/Switch:1*
T0*%
_class
loc:@dropout_1/cond/mul*
out_type0*
_output_shapes
:
Є
:training/RMSprop/gradients/dropout_1/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *%
_class
loc:@dropout_1/cond/mul*
valueB 
Л
Htraining/RMSprop/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/RMSprop/gradients/dropout_1/cond/mul_grad/Shape:training/RMSprop/gradients/dropout_1/cond/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@dropout_1/cond/mul
ѕ
6training/RMSprop/gradients/dropout_1/cond/mul_grad/MulMulFtraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Reshapedropout_1/cond/mul/y*(
_output_shapes
:џџџџџџџџџ*
T0*%
_class
loc:@dropout_1/cond/mul
І
6training/RMSprop/gradients/dropout_1/cond/mul_grad/SumSum6training/RMSprop/gradients/dropout_1/cond/mul_grad/MulHtraining/RMSprop/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*%
_class
loc:@dropout_1/cond/mul

:training/RMSprop/gradients/dropout_1/cond/mul_grad/ReshapeReshape6training/RMSprop/gradients/dropout_1/cond/mul_grad/Sum8training/RMSprop/gradients/dropout_1/cond/mul_grad/Shape*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0*(
_output_shapes
:џџџџџџџџџ
ў
8training/RMSprop/gradients/dropout_1/cond/mul_grad/Mul_1Muldropout_1/cond/mul/Switch:1Ftraining/RMSprop/gradients/dropout_1/cond/dropout/truediv_grad/Reshape*
T0*%
_class
loc:@dropout_1/cond/mul*(
_output_shapes
:џџџџџџџџџ
Ќ
8training/RMSprop/gradients/dropout_1/cond/mul_grad/Sum_1Sum8training/RMSprop/gradients/dropout_1/cond/mul_grad/Mul_1Jtraining/RMSprop/gradients/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
T0*%
_class
loc:@dropout_1/cond/mul*
_output_shapes
:*
	keep_dims( *

Tidx0

<training/RMSprop/gradients/dropout_1/cond/mul_grad/Reshape_1Reshape8training/RMSprop/gradients/dropout_1/cond/mul_grad/Sum_1:training/RMSprop/gradients/dropout_1/cond/mul_grad/Shape_1*
T0*%
_class
loc:@dropout_1/cond/mul*
Tshape0*
_output_shapes
: 
Л
#training/RMSprop/gradients/Switch_1Switchdense_1/Reludropout_1/cond/pred_id*
T0*
_class
loc:@dense_1/Relu*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
Њ
%training/RMSprop/gradients/Identity_1Identity#training/RMSprop/gradients/Switch_1*(
_output_shapes
:џџџџџџџџџ*
T0*
_class
loc:@dense_1/Relu
І
"training/RMSprop/gradients/Shape_2Shape#training/RMSprop/gradients/Switch_1*
T0*
_class
loc:@dense_1/Relu*
out_type0*
_output_shapes
:
Ж
(training/RMSprop/gradients/zeros_1/ConstConst&^training/RMSprop/gradients/Identity_1*
_class
loc:@dense_1/Relu*
valueB
 *    *
dtype0*
_output_shapes
: 
о
"training/RMSprop/gradients/zeros_1Fill"training/RMSprop/gradients/Shape_2(training/RMSprop/gradients/zeros_1/Const*
T0*
_class
loc:@dense_1/Relu*

index_type0*(
_output_shapes
:џџџџџџџџџ

Ctraining/RMSprop/gradients/dropout_1/cond/mul/Switch_grad/cond_gradMerge"training/RMSprop/gradients/zeros_1:training/RMSprop/gradients/dropout_1/cond/mul_grad/Reshape*
T0*
_class
loc:@dense_1/Relu*
N**
_output_shapes
:џџџџџџџџџ: 

!training/RMSprop/gradients/AddN_1AddNAtraining/RMSprop/gradients/dropout_1/cond/Switch_1_grad/cond_gradCtraining/RMSprop/gradients/dropout_1/cond/mul/Switch_grad/cond_grad*
T0*
_class
loc:@dense_1/Relu*
N*(
_output_shapes
:џџџџџџџџџ
Ц
5training/RMSprop/gradients/dense_1/Relu_grad/ReluGradReluGrad!training/RMSprop/gradients/AddN_1dense_1/Relu*
T0*
_class
loc:@dense_1/Relu*(
_output_shapes
:џџџџџџџџџ
т
;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad5training/RMSprop/gradients/dense_1/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes	
:

5training/RMSprop/gradients/dense_1/MatMul_grad/MatMulMatMul5training/RMSprop/gradients/dense_1/Relu_grad/ReluGraddense_1/kernel/read*
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul*
transpose_a( *(
_output_shapes
:џџџџџџџџџш
ћ
7training/RMSprop/gradients/dense_1/MatMul_grad/MatMul_1MatMuldense_1_input5training/RMSprop/gradients/dense_1/Relu_grad/ReluGrad*
transpose_a(* 
_output_shapes
:
ш*
transpose_b( *
T0*!
_class
loc:@dense_1/MatMul
w
&training/RMSprop/zeros/shape_as_tensorConst*
valueB"ш     *
dtype0*
_output_shapes
:
a
training/RMSprop/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ё
training/RMSprop/zerosFill&training/RMSprop/zeros/shape_as_tensortraining/RMSprop/zeros/Const*
T0*

index_type0* 
_output_shapes
:
ш

training/RMSprop/Variable
VariableV2*
shape:
ш*
shared_name *
dtype0*
	container * 
_output_shapes
:
ш
п
 training/RMSprop/Variable/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/zeros*
use_locking(*
T0*,
_class"
 loc:@training/RMSprop/Variable*
validate_shape(* 
_output_shapes
:
ш

training/RMSprop/Variable/readIdentitytraining/RMSprop/Variable*
T0*,
_class"
 loc:@training/RMSprop/Variable* 
_output_shapes
:
ш
g
training/RMSprop/zeros_1Const*
valueB*    *
dtype0*
_output_shapes	
:

training/RMSprop/Variable_1
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:
т
"training/RMSprop/Variable_1/AssignAssigntraining/RMSprop/Variable_1training/RMSprop/zeros_1*
T0*.
_class$
" loc:@training/RMSprop/Variable_1*
validate_shape(*
_output_shapes	
:*
use_locking(

 training/RMSprop/Variable_1/readIdentitytraining/RMSprop/Variable_1*
T0*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes	
:
y
(training/RMSprop/zeros_2/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"      
c
training/RMSprop/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
І
training/RMSprop/zeros_2Fill(training/RMSprop/zeros_2/shape_as_tensortraining/RMSprop/zeros_2/Const*
T0*

index_type0*
_output_shapes
:	

training/RMSprop/Variable_2
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:	*
shape:	
ц
"training/RMSprop/Variable_2/AssignAssigntraining/RMSprop/Variable_2training/RMSprop/zeros_2*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_2*
validate_shape(*
_output_shapes
:	
Ѓ
 training/RMSprop/Variable_2/readIdentitytraining/RMSprop/Variable_2*
T0*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes
:	
e
training/RMSprop/zeros_3Const*
valueB*    *
dtype0*
_output_shapes
:

training/RMSprop/Variable_3
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
с
"training/RMSprop/Variable_3/AssignAssigntraining/RMSprop/Variable_3training/RMSprop/zeros_3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_3

 training/RMSprop/Variable_3/readIdentitytraining/RMSprop/Variable_3*
_output_shapes
:*
T0*.
_class$
" loc:@training/RMSprop/Variable_3
b
 training/RMSprop/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
И
training/RMSprop/AssignAdd	AssignAddRMSprop/iterations training/RMSprop/AssignAdd/value*
use_locking( *
T0	*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 
x
training/RMSprop/mulMulRMSprop/rho/readtraining/RMSprop/Variable/read* 
_output_shapes
:
ш*
T0
[
training/RMSprop/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/RMSprop/subSubtraining/RMSprop/sub/xRMSprop/rho/read*
_output_shapes
: *
T0

training/RMSprop/SquareSquare7training/RMSprop/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
ш
w
training/RMSprop/mul_1Multraining/RMSprop/subtraining/RMSprop/Square*
T0* 
_output_shapes
:
ш
t
training/RMSprop/addAddtraining/RMSprop/multraining/RMSprop/mul_1* 
_output_shapes
:
ш*
T0
д
training/RMSprop/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/add*
use_locking(*
T0*,
_class"
 loc:@training/RMSprop/Variable*
validate_shape(* 
_output_shapes
:
ш

training/RMSprop/mul_2MulRMSprop/lr/read7training/RMSprop/gradients/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
ш
[
training/RMSprop/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
]
training/RMSprop/Const_1Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/RMSprop/clip_by_value/MinimumMinimumtraining/RMSprop/addtraining/RMSprop/Const_1*
T0* 
_output_shapes
:
ш

training/RMSprop/clip_by_valueMaximum&training/RMSprop/clip_by_value/Minimumtraining/RMSprop/Const*
T0* 
_output_shapes
:
ш
h
training/RMSprop/SqrtSqrttraining/RMSprop/clip_by_value*
T0* 
_output_shapes
:
ш
]
training/RMSprop/add_1/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
y
training/RMSprop/add_1Addtraining/RMSprop/Sqrttraining/RMSprop/add_1/y* 
_output_shapes
:
ш*
T0
~
training/RMSprop/truedivRealDivtraining/RMSprop/mul_2training/RMSprop/add_1*
T0* 
_output_shapes
:
ш
w
training/RMSprop/sub_1Subdense_1/kernel/readtraining/RMSprop/truediv*
T0* 
_output_shapes
:
ш
Т
training/RMSprop/Assign_1Assigndense_1/kerneltraining/RMSprop/sub_1*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(* 
_output_shapes
:
ш*
use_locking(
w
training/RMSprop/mul_3MulRMSprop/rho/read training/RMSprop/Variable_1/read*
T0*
_output_shapes	
:
]
training/RMSprop/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_2Subtraining/RMSprop/sub_2/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_1Square;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
v
training/RMSprop/mul_4Multraining/RMSprop/sub_2training/RMSprop/Square_1*
T0*
_output_shapes	
:
s
training/RMSprop/add_2Addtraining/RMSprop/mul_3training/RMSprop/mul_4*
T0*
_output_shapes	
:
з
training/RMSprop/Assign_2Assigntraining/RMSprop/Variable_1training/RMSprop/add_2*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_1*
validate_shape(*
_output_shapes	
:

training/RMSprop/mul_5MulRMSprop/lr/read;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
]
training/RMSprop/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training/RMSprop/Const_3Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_1/MinimumMinimumtraining/RMSprop/add_2training/RMSprop/Const_3*
_output_shapes	
:*
T0

 training/RMSprop/clip_by_value_1Maximum(training/RMSprop/clip_by_value_1/Minimumtraining/RMSprop/Const_2*
_output_shapes	
:*
T0
g
training/RMSprop/Sqrt_1Sqrt training/RMSprop/clip_by_value_1*
_output_shapes	
:*
T0
]
training/RMSprop/add_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *Пж3
v
training/RMSprop/add_3Addtraining/RMSprop/Sqrt_1training/RMSprop/add_3/y*
_output_shapes	
:*
T0
{
training/RMSprop/truediv_1RealDivtraining/RMSprop/mul_5training/RMSprop/add_3*
T0*
_output_shapes	
:
r
training/RMSprop/sub_3Subdense_1/bias/readtraining/RMSprop/truediv_1*
T0*
_output_shapes	
:
Й
training/RMSprop/Assign_3Assigndense_1/biastraining/RMSprop/sub_3*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:
{
training/RMSprop/mul_6MulRMSprop/rho/read training/RMSprop/Variable_2/read*
T0*
_output_shapes
:	
]
training/RMSprop/sub_4/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_4Subtraining/RMSprop/sub_4/xRMSprop/rho/read*
_output_shapes
: *
T0

training/RMSprop/Square_2Square7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
z
training/RMSprop/mul_7Multraining/RMSprop/sub_4training/RMSprop/Square_2*
T0*
_output_shapes
:	
w
training/RMSprop/add_4Addtraining/RMSprop/mul_6training/RMSprop/mul_7*
T0*
_output_shapes
:	
л
training/RMSprop/Assign_4Assigntraining/RMSprop/Variable_2training/RMSprop/add_4*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_2

training/RMSprop/mul_8MulRMSprop/lr/read7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
]
training/RMSprop/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training/RMSprop/Const_5Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_2/MinimumMinimumtraining/RMSprop/add_4training/RMSprop/Const_5*
T0*
_output_shapes
:	

 training/RMSprop/clip_by_value_2Maximum(training/RMSprop/clip_by_value_2/Minimumtraining/RMSprop/Const_4*
T0*
_output_shapes
:	
k
training/RMSprop/Sqrt_2Sqrt training/RMSprop/clip_by_value_2*
T0*
_output_shapes
:	
]
training/RMSprop/add_5/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
z
training/RMSprop/add_5Addtraining/RMSprop/Sqrt_2training/RMSprop/add_5/y*
T0*
_output_shapes
:	

training/RMSprop/truediv_2RealDivtraining/RMSprop/mul_8training/RMSprop/add_5*
T0*
_output_shapes
:	
x
training/RMSprop/sub_5Subdense_2/kernel/readtraining/RMSprop/truediv_2*
T0*
_output_shapes
:	
С
training/RMSprop/Assign_5Assigndense_2/kerneltraining/RMSprop/sub_5*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
v
training/RMSprop/mul_9MulRMSprop/rho/read training/RMSprop/Variable_3/read*
T0*
_output_shapes
:
]
training/RMSprop/sub_6/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_6Subtraining/RMSprop/sub_6/xRMSprop/rho/read*
_output_shapes
: *
T0

training/RMSprop/Square_3Square;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
v
training/RMSprop/mul_10Multraining/RMSprop/sub_6training/RMSprop/Square_3*
T0*
_output_shapes
:
s
training/RMSprop/add_6Addtraining/RMSprop/mul_9training/RMSprop/mul_10*
T0*
_output_shapes
:
ж
training/RMSprop/Assign_6Assigntraining/RMSprop/Variable_3training/RMSprop/add_6*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_3*
validate_shape(*
_output_shapes
:

training/RMSprop/mul_11MulRMSprop/lr/read;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
]
training/RMSprop/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training/RMSprop/Const_7Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_3/MinimumMinimumtraining/RMSprop/add_6training/RMSprop/Const_7*
_output_shapes
:*
T0

 training/RMSprop/clip_by_value_3Maximum(training/RMSprop/clip_by_value_3/Minimumtraining/RMSprop/Const_6*
T0*
_output_shapes
:
f
training/RMSprop/Sqrt_3Sqrt training/RMSprop/clip_by_value_3*
T0*
_output_shapes
:
]
training/RMSprop/add_7/yConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
u
training/RMSprop/add_7Addtraining/RMSprop/Sqrt_3training/RMSprop/add_7/y*
T0*
_output_shapes
:
{
training/RMSprop/truediv_3RealDivtraining/RMSprop/mul_11training/RMSprop/add_7*
T0*
_output_shapes
:
q
training/RMSprop/sub_7Subdense_2/bias/readtraining/RMSprop/truediv_3*
T0*
_output_shapes
:
И
training/RMSprop/Assign_7Assigndense_2/biastraining/RMSprop/sub_7*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
Д
training/group_depsNoOp	^loss/mul^metrics/acc/Mean^training/RMSprop/Assign^training/RMSprop/AssignAdd^training/RMSprop/Assign_1^training/RMSprop/Assign_2^training/RMSprop/Assign_3^training/RMSprop/Assign_4^training/RMSprop/Assign_5^training/RMSprop/Assign_6^training/RMSprop/Assign_7
0

group_depsNoOp	^loss/mul^metrics/acc/Mean

IsVariableInitializedIsVariableInitializeddense_1/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel

IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializeddense_2/bias*
dtype0*
_output_shapes
: *
_class
loc:@dense_2/bias

IsVariableInitialized_4IsVariableInitialized
RMSprop/lr*
dtype0*
_output_shapes
: *
_class
loc:@RMSprop/lr

IsVariableInitialized_5IsVariableInitializedRMSprop/rho*
_class
loc:@RMSprop/rho*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializedRMSprop/decay*
dtype0*
_output_shapes
: * 
_class
loc:@RMSprop/decay

IsVariableInitialized_7IsVariableInitializedRMSprop/iterations*%
_class
loc:@RMSprop/iterations*
dtype0	*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedtraining/RMSprop/Variable*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/RMSprop/Variable
Ђ
IsVariableInitialized_9IsVariableInitializedtraining/RMSprop/Variable_1*.
_class$
" loc:@training/RMSprop/Variable_1*
dtype0*
_output_shapes
: 
Ѓ
IsVariableInitialized_10IsVariableInitializedtraining/RMSprop/Variable_2*.
_class$
" loc:@training/RMSprop/Variable_2*
dtype0*
_output_shapes
: 
Ѓ
IsVariableInitialized_11IsVariableInitializedtraining/RMSprop/Variable_3*.
_class$
" loc:@training/RMSprop/Variable_3*
dtype0*
_output_shapes
: 
ж
initNoOp^RMSprop/decay/Assign^RMSprop/iterations/Assign^RMSprop/lr/Assign^RMSprop/rho/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign!^training/RMSprop/Variable/Assign#^training/RMSprop/Variable_1/Assign#^training/RMSprop/Variable_2/Assign#^training/RMSprop/Variable_3/Assign""

trainable_variablesє	ё	
\
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:08
M
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:08
\
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:08
M
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:08
R
RMSprop/lr:0RMSprop/lr/AssignRMSprop/lr/read:02RMSprop/lr/initial_value:08
V
RMSprop/rho:0RMSprop/rho/AssignRMSprop/rho/read:02RMSprop/rho/initial_value:08
^
RMSprop/decay:0RMSprop/decay/AssignRMSprop/decay/read:02RMSprop/decay/initial_value:08
r
RMSprop/iterations:0RMSprop/iterations/AssignRMSprop/iterations/read:02"RMSprop/iterations/initial_value:08
}
training/RMSprop/Variable:0 training/RMSprop/Variable/Assign training/RMSprop/Variable/read:02training/RMSprop/zeros:08

training/RMSprop/Variable_1:0"training/RMSprop/Variable_1/Assign"training/RMSprop/Variable_1/read:02training/RMSprop/zeros_1:08

training/RMSprop/Variable_2:0"training/RMSprop/Variable_2/Assign"training/RMSprop/Variable_2/read:02training/RMSprop/zeros_2:08

training/RMSprop/Variable_3:0"training/RMSprop/Variable_3/Assign"training/RMSprop/Variable_3/read:02training/RMSprop/zeros_3:08"Є	
cond_context		
и
dropout_1/cond/cond_textdropout_1/cond/pred_id:0dropout_1/cond/switch_t:0 *
dense_1/Relu:0
dropout_1/cond/dropout/Floor:0
dropout_1/cond/dropout/Shape:0
dropout_1/cond/dropout/add:0
dropout_1/cond/dropout/mul:0
5dropout_1/cond/dropout/random_uniform/RandomUniform:0
+dropout_1/cond/dropout/random_uniform/max:0
+dropout_1/cond/dropout/random_uniform/min:0
+dropout_1/cond/dropout/random_uniform/mul:0
+dropo