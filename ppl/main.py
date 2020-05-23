import cv2
import os
import time
import numpy as np
from filters import Filters


class Main:
    def __init__(self):
        # Main parameters
        # self.path = os.getcwd()
        # self.dataset_path = os.path.join(self.path, 'dataset')
        # self.output_path = os.path.join(self.path, 'out')

        self.dataset_path = '/media/marcos/Dados/Projects/Datasets/Exams'
        self.output_path = '/media/marcos/Dados/Projects/Results/Qualificacao/AlgoPPL/frames'
        self.labe_output_path = '/media/marcos/Dados/Projects/Results/Qualificacao/AlgoPPL/label'

        self.exams = ['benchmark_final.avi']
        # self.exams = os.listdir(self.dataset_path)

        self.detail_presentation = True
        self.save_output = True
        self.execute_invisible = True
        self.save_csv = True

        self.sleep_pause = 3

        self.filters = Filters(self.detail_presentation)

    def add_label(self, title, information):
        with open('{}/{}_label.csv'.format(self.labe_output_path, title), 'a', newline='') as file:
            file.write('{}\n'.format(information))
            file.close()

    def start_process(self):
        exam_jump_process = 0
        exam_process = 0
        for exam in self.exams:
            if exam_process >= exam_jump_process:
                name_file = exam.replace('.mp4', '').replace('.avi', '')
                csv_title = 'frame,center_x,center_y,radius,eye_size'
                self.add_label(name_file, csv_title)

                movie = cv2.VideoCapture("{}/{}".format(self.dataset_path, exam))
                self.pupil_process(movie, name_file)
            exam_process += 1

    def pupil_process(self, exam, name_file):
        number_frame = 0
        while True:
            _, frame = exam.read()

            if frame is None:
                break

            path_out = '/media/marcos/Dados/Projects/ProjectPupilometer/pupilometer/out'
            name_out = 'original_{}'.format(number_frame)
            file_out = '{}/{}.png'.format(path_out, name_out)
            cv2.imwrite(file_out, frame)

            center, radius, images = self.filters.pupil_analysis(frame, number_frame)
            final = np.copy(frame)

            final = cv2.circle(final, (center[0], center[1]), radius, (174, 174, 174), 2)

            if not self.execute_invisible:
                cv2.namedWindow('Training', cv2.WINDOW_NORMAL)
                cv2.imshow('Training', images['final'])

            if self.save_output:
                file_out = '{}/{}.png'.format(self.output_path, number_frame)
                cv2.imwrite(file_out, final)

            if self.save_csv:
                information = '{},{},{},{},{}'.format(number_frame, center[0], center[1], radius, 0)
                self.add_label(name_file, information)

            if cv2.waitKey(1) & 0xFF == ord('p'):  # Pause
                time.sleep(self.sleep_pause)

            number_frame += 1

        exam.release()
        cv2.destroyAllWindows


if __name__ == '__main__':
    main = Main()
    main.start_process()
