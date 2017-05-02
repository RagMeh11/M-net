'''
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python *.py

'''

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Graph, weighted_objective
from keras import objectives
from keras.objectives import mae, categorical_crossentropy, mse
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D, UpSampling2D, ZeroPadding3D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, SReLU

from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop,Adam
from keras.utils import np_utils, generic_utils
from six.moves import range
import keras
import numpy as np
import random
import pickle
import scipy.io as sio
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import h5py

###############################################################################

def my_to_categorical(y, nb_classes=None):
	Y = np.zeros([y.shape[0],y.shape[1],y.shape[2],y.max()+1],dtype='uint8')
	Y[np.nonzero(y)[0],np.nonzero(y)[1],np.nonzero(y)[2],y[np.nonzero(y)[0],np.nonzero(y)[1],np.nonzero(y)[2]]] = 1
	Y[np.where(y == 0)[0],np.where(y == 0)[1],np.where(y == 0)[2],y[np.where(y == 0)[0],np.where(y == 0)[1],np.where(y == 0)[2]]] = 1  
	return Y


nb_classes = 32
img_w = 256
img_h = 256

###########################################################################################################
graph = Graph()

graph.add_input(name='input', input_shape=(img_w,img_h,51,1))

print('------------------------\n\nInput Layers Added\n-------------------\n')

################################
graph.add_node(ZeroPadding3D(padding=(3, 3, 0), dim_ordering='tf'),name='ZP0',input='input')

graph.add_node(Convolution3D(16,7,7,51, border_mode='valid', activation='linear', init='glorot_uniform', dim_ordering='tf'), name='C3D', input='ZP0')
graph.add_node(Activation('relu'),name='AC3D',input='C3D')
graph.add_node(BatchNormalization(axis=-1),name='BNC3D',input='AC3D')

graph.add_node(Reshape((img_w,img_h,16),input_shape=(img_w,img_h,1,16)),input='BNC3D',name='ReShp')


graph.add_node(MaxPooling2D(pool_size=(2,2), border_mode='valid', dim_ordering='tf'), name='ND1', input='ReShp')

graph.add_node(MaxPooling2D(pool_size=(2,2), border_mode='valid', dim_ordering='tf'), name='ND2', input='ND1')

graph.add_node(MaxPooling2D(pool_size=(2,2), border_mode='valid', dim_ordering='tf'), name='ND3', input='ND2')


###############################

graph.add_node(Convolution2D(16,3,3, border_mode='same', activation='linear', init='glorot_uniform', dim_ordering='tf'), name='L11', input='ReShp')
graph.add_node(Activation('relu'),name='AL11',input='L11')
graph.add_node(BatchNormalization(axis=-1),name='BNL11',input='AL11')

graph.add_node(Convolution2D(32,3,3, border_mode='same', activation='linear', init='glorot_uniform' ,dim_ordering='tf'), name='L12', inputs=['BNL11','ReShp'])
graph.add_node(Activation('relu'),name='AL12',input='L12')
graph.add_node(BatchNormalization(axis=-1),name='BNL12',input='AL12')

graph.add_node(Dropout(0.3), input='BNL12', name='LD1')

#################################################
graph.add_node(MaxPooling2D(pool_size=(2,2), border_mode='valid', dim_ordering='tf'), name='LM1', input='LD1')
#################################################

graph.add_node(Convolution2D(32,3,3, border_mode='same', activation='linear', init='glorot_uniform', dim_ordering='tf'), name='L21', inputs=['LM1','ND1'])
graph.add_node(Activation('relu'),name='AL21',input='L21')
graph.add_node(BatchNormalization(axis=-1),name='BNL21',input='AL21')

graph.add_node(Convolution2D(48,3,3, border_mode='same', activation='linear', init='glorot_uniform' ,dim_ordering='tf'), name='L22', inputs=['BNL21','LM1','ND1'])
graph.add_node(Activation('relu'),name='AL22',input='L22')
graph.add_node(BatchNormalization(axis=-1),name='BNL22',input='AL22')


graph.add_node(Dropout(0.3), input='BNL22', name='LD2')

#########################################################
graph.add_node(MaxPooling2D(pool_size=(2,2), border_mode='valid', dim_ordering='tf'), name='LM2', input='LD2')
########################################################

graph.add_node(Convolution2D(48,3,3, border_mode='same', activation='linear', init='glorot_uniform', dim_ordering='tf'), name='L31', inputs=['LM2','ND2'])
graph.add_node(Activation('relu'),name='AL31',input='L31')
graph.add_node(BatchNormalization(axis=-1),name='BNL31',input='AL31')


graph.add_node(Convolution2D(64,3,3, border_mode='same', activation='linear', init='glorot_uniform' ,dim_ordering='tf'), name='L32', inputs=['BNL31','LM2','ND2'])
graph.add_node(Activation('relu'),name='AL32',input='L32')
graph.add_node(BatchNormalization(axis=-1),name='BNL32',input='AL32')


graph.add_node(Dropout(0.3), input='BNL32', name='LD3')

##############################################################
graph.add_node(MaxPooling2D(pool_size=(2,2), border_mode='valid', dim_ordering='tf'), name='LM3', input='LD3')
############################################################

graph.add_node(Convolution2D(64,3,3, border_mode='same', activation='linear', init='glorot_uniform', dim_ordering='tf'), name='L41', inputs=['LM3','ND3'])
graph.add_node(Activation('relu'),name='AL41',input='L41')
graph.add_node(BatchNormalization(axis=-1),name='BNL41',input='AL41')


graph.add_node(Convolution2D(128,3,3, border_mode='same', activation='linear', init='glorot_uniform' ,dim_ordering='tf'), name='N1', inputs=['BNL41','LM3','ND3'])
graph.add_node(Activation('relu'),name='AN1',input='N1')
graph.add_node(BatchNormalization(axis=-1),name='BNN1',input='AN1')


graph.add_node(Convolution2D(64,3,3, border_mode='same', activation='linear', init='glorot_uniform' ,dim_ordering='tf'), name='R41', input='BNN1')
graph.add_node(Activation('relu'),name='AR41',input='R41')
graph.add_node(BatchNormalization(axis=-1),name='BNR41',input='AR41')


graph.add_node(Dropout(0.3), input='BNR41', name='RD3')

#################################################################
graph.add_node(UpSampling2D(size=(2, 2), dim_ordering='tf'), name='RU3' , input='RD3' )
##############################################################

graph.add_node(Convolution2D(64,3,3, border_mode='same', activation='linear', init='glorot_uniform', dim_ordering='tf'), name='R32', inputs=['RU3','LD3','ND2'])
graph.add_node(Activation('relu'),name='AR32',input='R32')
graph.add_node(BatchNormalization(axis=-1),name='BNR32',input='AR32')


graph.add_node(Convolution2D(48,3,3, border_mode='same', activation='linear', init='glorot_uniform' ,dim_ordering='tf'), name='R31', inputs=['BNR32','RU3','LD3','ND2'])
graph.add_node(Activation('relu'),name='AR31',input='R31')
graph.add_node(BatchNormalization(axis=-1),name='BNR31',input='AR31')


graph.add_node(Dropout(0.3), input='BNR31', name='RD2')

###########################################################
graph.add_node(UpSampling2D(size=(2, 2), dim_ordering='tf'), name='RU2' , input='RD2' )
#############################################################

graph.add_node(Convolution2D(48,3,3, border_mode='same', activation='linear', init='glorot_uniform', dim_ordering='tf'), name='R22', inputs=['RU2','LD2','ND1'])
graph.add_node(Activation('relu'),name='AR22',input='R22')
graph.add_node(BatchNormalization(axis=-1),name='BNR22',input='AR22')


graph.add_node(Convolution2D(32,3,3, border_mode='same', activation='linear', init='glorot_uniform' ,dim_ordering='tf'), name='R21', inputs=['BNR22','RU2','LD2','ND1'])
graph.add_node(Activation('relu'),name='AR21',input='R21')
graph.add_node(BatchNormalization(axis=-1),name='BNR21',input='AR21')


graph.add_node(Dropout(0.3), input='BNR21', name='RD1')

###########################################################
graph.add_node(UpSampling2D(size=(2, 2), dim_ordering='tf'), name='RU1' , input='RD1' )
##############################################

graph.add_node(Convolution2D(32,3,3, border_mode='same', activation='linear', init='glorot_uniform', dim_ordering='tf'), name='R12', inputs=['RU1','LD1','ReShp'])
graph.add_node(Activation('relu'),name='AR12',input='R12')
graph.add_node(BatchNormalization(axis=-1),name='BNR12',input='AR12')


graph.add_node(Convolution2D(64,3,3, border_mode='same', activation='linear', init='glorot_uniform' ,dim_ordering='tf'), name='R11', inputs=['BNR12','RU1','LD1','ReShp'])
graph.add_node(Activation('relu'),name='AR11',input='R11')
graph.add_node(BatchNormalization(axis=-1),name='BNR11',input='AR11')


graph.add_node(Dropout(0.3), input='BNR11', name='RD01')
#####################################################################

graph.add_node(UpSampling2D(size=(2, 2), dim_ordering='tf'), name='RU2_2' , inputs=['RU2','LD2','ND1'] )
graph.add_node(UpSampling2D(size=(4, 4), dim_ordering='tf'), name='RU3_2' , inputs=['RU3','LD3','ND2'] )
######################################################################
graph.add_node(Convolution2D(200,3,3, border_mode='same', activation='relu', init='glorot_uniform' ,dim_ordering='tf'), name='R00', inputs=['RD01','RU1','RU2_2','RU3_2','LD1','ReShp'])
graph.add_node(Activation('relu'),name='AR00',input='R00')
graph.add_node(BatchNormalization(axis=-1),name='BNR00',input='AR00')


graph.add_node(Dropout(0.3), input='BNR00', name='RD02')


graph.add_node(Convolution2D(32,1,1, border_mode='same', activation='linear', init='glorot_uniform' ,dim_ordering='tf'), name='LC', input='RD02')
graph.add_node(Activation('softmax'),name='SM',input='LC')

graph.add_output(name='output', input='SM')


print('------------------------\n\nSGD\n-------------------\n')
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
adm = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
graph.compile(optimizer=adm, loss={'output':'categorical_crossentropy'})



b_size = 8

main_path ='./IBSR18_slices/' 

######################################################
class_im = sio.loadmat(main_path+'Data/training/total_class_imbalance.mat')
hist = class_im['total_hist'].squeeze()


hist = hist.astype('float32')
print(hist.shape)
ss=sum(hist).astype('float32')
print(ss)
print(hist[0])
class_weight = dict([(i, ss/hist[i]) for i in range(nb_classes)])
print(class_weight)

############################################################################

training_data = np.load(main_path+'Data/training/npz/whole.npz')
training_inp = training_data['img'].astype('float32')
training_gt = training_data['gt'].astype('uint8')
training_inp = np.transpose(training_inp)
training_inp = training_inp.reshape(training_inp.shape+(1,)).astype('float32')
training_gt = np.transpose(training_gt)

print(training_inp.shape)
print(training_gt.shape)

training_gt_category = my_to_categorical(training_gt, nb_classes)
print('training GT after categorical')
print(training_gt_category.shape)


if(np.all(training_gt == np.argmax(training_gt_category,axis=-1))):
	print('Proper Conversion to categorical')

###################################################################################

for ep in range(weig_load+1,no_ep+1):
	print('\n\n###########################################')
	print('#########################################3\n\n')

	print('Training Model:\n\n')

	print('Epoch: %d' %(ep))
    

        MCP=keras.callbacks.ModelCheckpoint(main_path+'CNN_output/weights/weights-%d.hdf5'%(ep), monitor='val_loss',save_best_only=False)	
		

	
	graph.fit({'input':training_inp,'output':training_gt_category}, batch_size=b_size,  verbose=1 ,nb_epoch=1, callbacks = [MCP], class_weight=class_weight)
	
	
######################################################################################################
	
	if((ep % ep_lap) == 0):

		print('\n\n-----------------------------------------\n\n')
		print('Training Accuracy:')
		t_dice = 0.0
		t_acc = 0.0
		scr =0.0	
		
		for trai_imgn in range(tr_images):
			print('\n\nloading image %d for Accuracy\n'% (trai_imgn+1))
			
			npz_contents = np.load(main_path+'Data/training/npz/%d.npz'%(trai_imgn+1))
			print('data loaded')
			trai_inp = npz_contents['img'].astype('float32')
			
			trai_gt = npz_contents['gt'].astype('uint8')
            		trai_inp = np.transpose(trai_inp)
            		trai_inp = trai_inp.reshape(trai_inp.shape+(1,)).astype('float32')
            		trai_gt = np.transpose(trai_gt)
			print (trai_inp.shape)
			
		   	prediction = graph.predict({'input':trai_inp}, batch_size=b_size)

	        	trai_pre = np.argmax(prediction['output'],axis=-1).astype('uint8')
	        	
	        	trai_pre = np.reshape(trai_pre,trai_pre.shape[0]*trai_pre.shape[1]*trai_pre.shape[2]).astype('uint8')
	        	trai_gt = np.reshape(trai_gt,trai_gt.shape[0]*trai_gt.shape[1]*trai_gt.shape[2]).astype('uint8')

	        	[my_accu,zero_accu] = accuracy(trai_pre, trai_gt)
	        	skl_dice = f1_score(trai_gt, trai_pre,average='macro')
	        	skl_accu = accuracy_score(trai_gt, trai_pre)

			print ('skl accu = ',skl_accu,'skl dice coeff = ',skl_dice,'zero accu = ',zero_accu, 'my_accu = ', my_accu)
			trai_pre = np.reshape(trai_pre,[trai_inp.shape[0],trai_inp.shape[1],trai_inp.shape[2]]).astype('uint8')
	        	score = 0
	        	t_dice+=skl_dice
	        	t_acc += skl_accu
			scr=scr+score
	        	fp1 = open(log_file,'a')
	        	fp1.write('epoch:%d image:%d Training accuracy:%f Dice Coeff:%f\n'%(ep,trai_imgn,skl_accu,skl_dice))
	        	fp1.close()
			

	    	scr = scr/tr_images
	    	t_dice = t_dice/tr_images
	    	t_acc = t_acc/tr_images
	    	print('\n\nTraining Overall Dice Coeffient: %f'%(t_dice))
	    	print('Training Overall Accuracy: %f'%(t_acc))
	    	print('Training Score: %f'%(scr))
			

	######################################################################################################
		print('\n-----------------------------------------\n\n')
		print('validation Accuracy:')
		t_dice = 0.0
		t_acc = 0.0
		scr =0.0	
		
		for valid_imgn in range(valid_images):
			print('\nloading image %d for Accuracy\n'% (valid_imgn+1))
			npz_contents = np.load(main_path+'Data/validation/npz/%d.npz'%(valid_imgn+1))
			print('data loaded')
			valid_inp = npz_contents['img'].astype('float32')
			valid_gt = npz_contents['gt'].astype('uint8')
			valid_inp = np.transpose(valid_inp)
			valid_inp = valid_inp.reshape(valid_inp.shape+(1,)).astype('float32')
            		valid_gt = np.transpose(valid_gt)
            
			print (valid_inp.shape)
			
		   	prediction = graph.predict({'input':valid_inp}, batch_size=b_size)

	        	
	        	valid_pre = np.argmax(prediction['output'],axis=-1).astype('uint8')

	        	valid_pre = np.reshape(valid_pre,valid_pre.shape[0]*valid_pre.shape[1]*valid_pre.shape[2]).astype('uint8')
	        	valid_gt = np.reshape(valid_gt,valid_gt.shape[0]*valid_gt.shape[1]*valid_gt.shape[2]).astype('uint8')

	        	[my_accu,zero_accu] = accuracy(valid_pre, valid_gt)
	        	skl_dice = f1_score(valid_gt, valid_pre,average='macro')
	        	skl_accu = accuracy_score(valid_gt, valid_pre)

			print ('\nskl accu = ',skl_accu,'skl dice coeff = ',skl_dice,'zero accu = ',zero_accu,'my_accu = ',my_accu)
			valid_pre = np.reshape(valid_pre,[valid_inp.shape[0],valid_inp.shape[1],valid_inp.shape[2]]).astype('uint8')
	        	score = 0
	        	t_dice+=skl_dice
	        	t_acc += skl_accu
			scr=scr+score
	        	fp1 = open(log_file,'a')
	        	fp1.write('epoch:%d image:%d validation accuracy:%f Dice Coeff:%f\n'%(ep,valid_imgn,skl_accu,skl_dice))
	        	fp1.close()
			

	    	scr = scr/valid_images
	    	t_dice = t_dice/valid_images
	    	t_acc = t_acc/valid_images
	    	print('\n\nvalidation Overall Dice Coeffient: %f'%(t_dice))
	    	print('validation Overall Accuracy: %f'%(t_acc))
	    	print('validation Score: %f'%(scr))
			
