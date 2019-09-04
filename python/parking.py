import yaml
import numpy as np
import cv2
import time
import sys

sys.setrecursionlimit(15000)

totallibre=0
totalocupado=0

def captura():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while img_counter==0:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        img_name = "captura.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

def total():
    print ("Libre: "+str(totallibre))
    print("Ocupado: "+str(totalocupado))
    return totallibre, totalocupado

class clase:
    def pk(self):
        global totallibre
        global totalocupado
        capture_duration = 4
        # Capturar video
        cap = cv2.VideoCapture(0)

        # Revisar si la camara se ha abierto correctamente
        if (cap.isOpened() == False):
            print("No se puede usar la camara")

        # Recuperar resolucion camara
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # imprime la resolucion del video
        print("------------------------")
        print("\nResolución de video:")
        print(cap.get(3))
        print(cap.get(4))
        print("------------------------")

        # Definir el codec
        out = cv2.VideoWriter('../datasets/3.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                              (frame_width, frame_height))

        while (True):
            start_time = time.time()
            while (int(time.time() - start_time) < capture_duration):
                ret, frame = cap.read()

                if ret == True:

                    out.write(frame)

                    # Mostrar resultado
                    cv2.imshow('frame', frame)

                    # Pulsar "Q" para detener la grabacion
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            # Parar el bucle
            else:
                break

            # si sale bien:
        cap.release()
        out.release()

        # cerrar todos los frames
        cv2.destroyAllWindows()

        fn = r"../datasets/3.avi"
        fn_yaml = r"../datasets/parking2.yml"
        config = {'save_video': False,
                  'text_overlay': True,
                  'parking_overlay': True,
                  'parking_id_overlay': False,
                  'parking_detection': True,
                  'min_area_motion_contour': 60,
                  'park_sec_to_wait': 3,
                  'start_frame': 0}

        # Seleccionar el archivo de video
        cap = cv2.VideoCapture(fn)
        # recupera fotogramas, tamaño,etc..
        video_info = {'fps': cap.get(cv2.CAP_PROP_FPS),
                      'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
                      'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        cap.set(cv2.CAP_PROP_POS_FRAMES, config['start_frame'])  # jump to frame

        # Definir el codec, tamaño, etc..
        if config['save_video']:
            fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')
            out = cv2.VideoWriter(fn_out, -1, 25.0,
                                  (video_info['width'], video_info['height']))

        # Leer los poligonos
        with open(fn_yaml, 'r') as stream:
            parking_data = yaml.load(stream)
        parking_contours = []
        parking_bounding_rects = []
        parking_mask = []
        for park in parking_data:
            points = np.array(park['points'])
            rect = cv2.boundingRect(points)
            points_shifted = points.copy()
            points_shifted[:, 0] = points[:, 0] - rect[0]
            points_shifted[:, 1] = points[:, 1] - rect[1]
            parking_contours.append(points)
            parking_bounding_rects.append(rect)
            mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                                    color=255, thickness=-1, lineType=cv2.LINE_8)
            mask = mask == 255
            parking_mask.append(mask)

        parking_status = [False] * len(parking_data)
        parking_buffer = [None] * len(parking_data)

        while (cap.isOpened()):

            # Leer frame por frame
            video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            video_cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            if ret == False:
                # al terminar, mostrar el estado del parking:
                print("----------------------------------")
                print ("PROCESO TERMINADO")
                print("------------------------")
                totallibre=spot
                totalocupado=occupied


                break
            spot = 0
            occupied = 0
            frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
            frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
            frame_out = frame.copy()

            if config['parking_detection']:
                for ind, park in enumerate(parking_data):
                    points = np.array(park['points'])
                    rect = parking_bounding_rects[ind]
                    roi_gray = frame_gray[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]

                    points[:, 0] = points[:, 0] - rect[0]
                    points[:, 1] = points[:, 1] - rect[1]
                    status = np.std(roi_gray) < 22 and np.mean(roi_gray) > 53
                    if status != parking_status[ind] and parking_buffer[ind] == None:
                        parking_buffer[ind] = video_cur_pos
                    elif status != parking_status[ind] and parking_buffer[ind] != None:
                        if video_cur_pos - parking_buffer[ind] > config['park_sec_to_wait']:
                            parking_status[ind] = status
                            parking_buffer[ind] = None
                    elif status == parking_status[ind] and parking_buffer[ind] != None:
                        parking_buffer[ind] = None

            if config['parking_overlay']:
                for ind, park in enumerate(parking_data):
                    points = np.array(park['points'])
                    if parking_status[ind]:
                        color = (0, 255, 0)
                        spot = spot + 1
                    else:
                        color = (0, 0, 255)
                        occupied = occupied + 1
                    cv2.drawContours(frame_out, [points], contourIdx=-1,
                                     color=color, thickness=2, lineType=cv2.LINE_8)
                    moments = cv2.moments(points)
                    centroid = (int(moments['m10'] / moments['m00']) - 3, int(moments['m01'] / moments['m00']) + 3)
                    cv2.putText(frame_out, str(park['id']), (centroid[0] + 1, centroid[1] + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame_out, str(park['id']), (centroid[0] - 1, centroid[1] - 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame_out, str(park['id']), (centroid[0] + 1, centroid[1] - 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame_out, str(park['id']), (centroid[0] - 1, centroid[1] + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame_out, str(park['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                                cv2.LINE_AA)

            # Texto
            if config['text_overlay']:
                str_on_frame = "Libres: %d Ocupados: %d" % (spot, occupied)
                cv2.putText(frame_out, str_on_frame, (5, 90), cv2.FONT_HERSHEY_TRIPLEX,
                            0.7, (0, 0, 0), 2, cv2.LINE_AA)

                str_on_frame = "Frames: %d/%d" % (video_cur_frame, video_info['num_of_frames'])
                cv2.putText(frame_out, str_on_frame, (5, 30), cv2.FONT_HERSHEY_TRIPLEX,
                            0.7, (0, 0, 0), 2, cv2.LINE_AA)

            # si se quiere guardar el video (desactivado)
            if config['save_video']:
                if video_cur_frame % 35 == 0:
                    out.write(frame_out)

            # Mostrar video
            cv2.imshow('Deteccion de parking', frame_out)
            cv2.waitKey(40)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            elif k == ord('c'):
                cv2.imwrite('frame%d.jpg' % video_cur_frame, frame_out)
            elif k == ord('j'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_cur_frame + 1000)


        cap.release()
        if config['save_video']: out.release()
        cv2.destroyAllWindows()

        return