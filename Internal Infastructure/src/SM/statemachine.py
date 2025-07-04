from transitions import Machine

from models import cnn_model
from models import lstm_model
from vision import hsv_filter
from vision import video_utils
from data import data_loader

class PitchRecognitionSM:
    states = ['idle', 'video_received', 'preprocessing', 'feature_extraction', 'classification', 'result_output', 'error_handling']

    def __init__(self):
        self.machine = Machine(model=self, states=PitchRecognitionSM.states, initial='idle')
        self.machine.add_transition(trigger='upload', source='idle', dest='video_received', after='validate_video')
        self.machine.add_transition(trigger='preprocess', source='video_received', dest='preprocessing', after='extract_frames_and_filter')
        self.machine.add_transition(trigger='extract_features', source='preprocessing', dest='feature_extraction', after='run_cnn')
        self.machine.add_transition(trigger='classify', source='feature_extraction', dest='classification', after='run_lstm')
        self.machine.add_transition(trigger='output_result', source='classification', dest='result_output', after='send_result')
        self.machine.add_transition(trigger='reset', source='result_output', dest='idle')

        self.machine.add_transition(trigger='fail', source='*', dest='error_handling', after='handle_error')
        self.machine.add_transition(trigger='retry', source='error_handling', dest='idle')

    def validate_video(self):
        # check video file validity
        pass

    def extract_frames_and_filter(self):
        # HSV filter and frame extraction
        pass

    def run_cnn(self):
        # pass frames through CNN
        pass

    def run_lstm(self):
        # pass CNN features through LSTM
        pass

    def send_result(self):
        # send output to user
        pass

    def handle_error(self):
        # log error and decide on retry or abort
        pass


if __name__ == "__main__":

    sm = PitchRecognitionSM()

    try:
        done = False
        while not done:
            sm.validate_video()
            sm.extract_frames_and_filter()
            sm.run_cnn()
            sm.run_lstm()
            sm.send_result()
            done = True

    except KeyboardInterrupt:
        sm.handle_error()
