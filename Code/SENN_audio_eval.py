"""
Evaluate a trained model using a noisy speech
"""
import fnmatch
import os

import librosa
import numpy as np
import tensorflow as tf
from numpy.lib import stride_tricks

from Code import SENN


def STFT(sig, frameSize, overlapFac=0.75, window=np.hanning):
    """ short time fourier transform of audio signal """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    # samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
    samples = np.array(sig, dtype='float64')
    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


FLAGS = tf.app.flags.FLAGS
# directory to load the model
tf.app.flags.DEFINE_string(
    'train_dir',
    '/Users/Future/Desktop/Summer-2018/Research-Labs/DML/CNN-for-single-channel-speech-enhancement/Results/event_logs/',
    """Directory where to write event logs """
    """and checkpoint.""")

# meaning of the following params can be found in SENN_train.py
LR = 0.01
N_IN = 8
NEFF = 129
NFFT = 256
N_OUT = 1
Overlap = 0.75
mul_fac = 0.2
NMOVE = int((1 - Overlap) * NFFT)
audionames = []
audio_path = '/Users/Future/Desktop/Summer-2018/Research-Labs/DML/CNN-for-single-channel-speech-enhancement/Data/TIMIT/TEST/DR1/FAKS0/'

for root, dirs, files in os.walk(audio_path):
    for file in files:
        if fnmatch.fnmatch(file, '*.WAV'):
            audionames.append(file.split('.')[0])
        elif fnmatch.fnmatch(file, '*.wav'):
            audionames.append(file.split('.')[0])

print(audionames)
for audioname in audionames:

    audio_dir = audio_path + audioname + '.WAV'
    noise_dir = '/Users/Future/Desktop/Summer-2018/Research-Labs/DML/CNN-for-single-channel-speech-enhancement/Data/Noise/train/noise-1.wav'
    # dir to write the clean speech
    out_org_dir = '/Users/Future/Desktop/Summer-2018/Research-Labs/DML/CNN-for-single-channel-speech-enhancement/Results/Voices/' + audioname + '_withnoise.wav'
    # dir to write the clean speech inference
    out_audio_dir = '/Users/Future/Desktop/Summer-2018/Research-Labs/DML/CNN-for-single-channel-speech-enhancement/Results/Voices/' + audioname + '_denoised.wav'

    audio_org, audio_sr = librosa.load(audio_dir, sr=None)
    print(audio_sr)
    noise_org, noise_sr = librosa.load(noise_dir, sr=None)
    print(noise_sr)

    audio_len = len(audio_org)
    noise_len = len(noise_org)
    tot_len = max(audio_len, noise_len)

    # mix the sample
    if audio_len < noise_len:
        rep_time = int(np.floor(noise_len / audio_len))
        left_len = noise_len - audio_len * rep_time
        temp_data = np.tile(audio_org, [1, rep_time])
        temp_data.shape = (temp_data.shape[1],)
        audio = np.hstack((
            temp_data, audio_org[:left_len]))
        noise = np.array(noise_org)
    else:
        rep_time = int(np.floor(audio_len / noise_len))
        left_len = audio_len - noise_len * rep_time
        temp_data = np.tile(noise_org, [1, rep_time])
        temp_data.shape = (temp_data.shape[1],)
        noise = np.hstack((
            temp_data, noise_org[:left_len]))
        audio = np.array(audio_org)

    in_audio = (audio + mul_fac * noise)

    in_stft = STFT(in_audio, NFFT, Overlap)
    in_stft_amp = np.maximum(np.abs(in_stft), 1e-5)
    in_data = 20. * np.log10(in_stft_amp * 100)
    # ipdb.set_trace()
    phase_data = in_stft / in_stft_amp

    # ipdb.set_trace()
    data_len = in_data.shape[0]
    assert NEFF == in_data.shape[1], 'Uncompatible image height'
    out_len = data_len - N_IN + 1
    out_audio = np.zeros(shape=[int((out_len - 1) * NMOVE + NFFT)])

    init_op = tf.initialize_all_variables()

    # with tf.Graph().as_default():

    # construct the Net, meaning of the ops can be found in SENN_train.py

    batch_size = 1

    SE_Net = SENN.SE_NET(
        batch_size, NEFF, N_IN, N_OUT)

    images = tf.placeholder(tf.float32, [N_IN, NEFF])

    targets = tf.placeholder(tf.float32, [NEFF])

    inf_targets = SE_Net.inference(images, is_train=False)

    loss = SE_Net.loss(inf_targets, targets)

    # train_op = SE_Net.train(loss, LR)

    saver = tf.train.Saver(tf.all_variables())

    # summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
        # restore the model
        saver.restore(sess,
                      '/Users/Future/Desktop/Summer-2018/Research-Labs/DML/CNN-for-single-channel-speech-enhancement/Results/event_logs/Old-3/model.ckpt-17500')
        print("Model restored")
        # sess.run(tf.initialize_all_variables())
        i = 0
        while i < out_len:
            # show progress
            if i % 100 == 0:
                print('frame num: %d' % (i))
            feed_in_data = in_data[i:i + N_IN][:]
            # normalization
            data_mean = np.mean(feed_in_data)
            data_var = np.var(feed_in_data)
            feed_in_data = (feed_in_data - data_mean) / np.sqrt(data_var)
            # get the speech inference
            inf_frame, = sess.run(
                [inf_targets],
                feed_dict={images: feed_in_data})
            inf_frame = inf_frame * np.sqrt(data_var) + data_mean
            out_amp_tmp = 10 ** (inf_frame / 20) / 100
            out_stft = out_amp_tmp * phase_data[i + N_IN - 1][:]
            out_stft.shape = (NEFF,)
            con_data = out_stft[-2:0:-1].conjugate()
            out_amp = np.concatenate((out_stft, con_data))
            frame_out_tmp = np.fft.ifft(out_amp).astype(np.float64)
            # frame_out_tmp = frame_out_tmp / 255
            # overlap and add to get the final time domain wavform
            out_audio[i * NMOVE: i * NMOVE + NFFT] += frame_out_tmp * 0.5016
            # ipdb.set_trace()
            i = i + 1
        # length = img.shape[]

    # ipdb.set_trace()
    # store the computed results
    librosa.output.write_wav(out_audio_dir, out_audio, audio_sr)
    librosa.output.write_wav(out_org_dir, in_audio, audio_sr)

    tf.reset_default_graph()
