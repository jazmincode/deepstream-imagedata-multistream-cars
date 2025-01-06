#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys

sys.path.append('../')
import gi
import configparser

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import time
import sys
import math
import platform
from common.platform_info import PlatformInfo
from common.bus_call import bus_call
from common.FPS import PERF_DATA
import numpy as np
import random
import pyds
import cv2





from draw import *
from utils.utils import *
from utils.image_utils import *




perf_data = None
frame_count = {}
saved_count = {}

general_roi = []
up_roi = []
down_roi = []
down = []
up = []
track_history = []
track_history_count = []




global PGIE_CLASS_ID_VEHICLE
PGIE_CLASS_ID_VEHICLE = 0
global PGIE_CLASS_ID_PERSON
PGIE_CLASS_ID_PERSON = 2

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
pgie_classes_str = ["Vehicle", "TwoWheeler", "Person", "RoadSign"]

MIN_CONFIDENCE = 0.3
MAX_CONFIDENCE = 0.4

frames_with_draw = []

ROI = np.array([[2,1060], [1915,1050], [1915,350],[2,350]], np.int32).reshape((-1,1,2))




#------------------------------------------------------------------

# tiler_sink_pad_buffer_probe  will extract metadata received on tiler src pad
# and update params for drawing rectangle, object information etc.
def tiler_sink_pad_buffer_probe(pad, info, u_data):
    global frames_with_draw
    global up
    global down
    global up_roi
    global down_roi
    global track_history
    global track_history_count

    counters = (general_roi, up, down, up_roi, down_roi)

 
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list

    #inicializando frame_copy
    frame_copy = np.zeros((1080, 1920, 4), dtype=np.uint8)

    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        is_first_obj = True
        save_image = False
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0
        }
        
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            # Periodically check for objects with borderline confidence value that may be false positive detections.
            # If such detections are found, annotate the frame with bboxes and confidence value.
            # Save the annotated frame to file.

            #---------------Login new------------------------------------------
            
            

            index_track=get_index(track_history,str(obj_meta.object_id))
            if obj_meta.class_id == PGIE_CLASS_ID_VEHICLE and index_track == -1:
                track_history.append({str(obj_meta.object_id): []})
                
                
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            orientation = get_direction(track_history, obj_meta.rect_params, obj_meta.object_id)
                
            n_frame = draw_bounding_boxes(n_frame, obj_meta, obj_meta.confidence, orientation, ROI)
            n_frame = draw_trajectories(n_frame, track_history, obj_meta)
            n_frame = draw_counts(n_frame, counters)
            frame_copy = np.array(n_frame, copy=True, order='C')
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                
                
                
            if (orientation == 'Up' or orientation == 'Down'):
                x1,y1,x2,y2 = convert_box_format(obj_meta.rect_params.left, obj_meta.rect_params.top, obj_meta.rect_params.width, obj_meta.rect_params.height)
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                if(is_inside_roi(points, ROI)):
                    cropped_frame = crop_frame(frame_copy, obj_meta)
                    path = 'frames_'+orientation
                    save_frame(cropped_frame, path)


        
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        #after draw every object save in the list frames_with_draw
        path = 'frames_video'
        save_frame(frame_copy, path)
        
        
        
        

       
        # update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)


        saved_count["stream_{}".format(frame_meta.pad_index)] += 1

        

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    
 
    #past tracking meta data
    l_user=batch_meta.batch_user_meta_list
    
    
    while l_user is not None:
        
        try:
            # Note that l_user.data needs a cast to pyds.NvDsUserMeta
            # The casting is done by pyds.NvDsUserMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone
            user_meta=pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if(user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META):
            try:
                # Note that user_meta.user_meta_data needs a cast to pyds.NvDsTargetMiscDataBatch
                # The casting is done by pyds.NvDsTargetMiscDataBatch.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone
                pPastDataBatch = pyds.NvDsTargetMiscDataBatch.cast(user_meta.user_meta_data)
            except StopIteration:
                break
            for miscDataStream in pyds.NvDsTargetMiscDataBatch.list(pPastDataBatch):
                print("streamId=",miscDataStream.streamID)
                #print("surfaceStreamID=",miscDataStream.surfaceStreamID)
                #print("numAllocated=",miscDataStream.numAllocated)
                #print("numFilled=",miscDataStream.numFilled)
                for miscDataObj in pyds.NvDsTargetMiscDataStream.list(miscDataStream):
                    print("numobj=",miscDataObj.numObj)
                    print("uniqueId=",miscDataObj.uniqueId)

                    index_track=get_index(track_history_count,str(miscDataObj.uniqueId))
                    if miscDataObj.classId == PGIE_CLASS_ID_VEHICLE and index_track == -1:
                        track_history_count.append({str(miscDataObj.uniqueId): []})
                     
                    #por cada objeto detectado este for recorre el historico 
                    #a lo largo de todo el video
                    
                    for miscDataFrame in pyds.NvDsTargetMiscDataObject.list(miscDataObj):
                        #print("uniqmiscDataFrame.tBbox.left=",miscDataFrame.tBbox.left)
                        
                        if (miscDataObj.classId == PGIE_CLASS_ID_VEHICLE):

                            x1,y1,x2,y2 = convert_box_format(miscDataFrame.tBbox.left, miscDataFrame.tBbox.top, miscDataFrame.tBbox.width, miscDataFrame.tBbox.height)
                            bbox_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

                            
                            orientation = get_direction(track_history_count, miscDataFrame.tBbox, miscDataObj.uniqueId)

                            
                            
                            if  miscDataFrame.age > 2:
                                if orientation == 'Up' and  not(miscDataObj.uniqueId in up):
                                    up.append(miscDataObj.uniqueId)
                                    update_trend(miscDataObj.uniqueId,up,down)
                        
                                elif orientation == 'Down' and  not(miscDataObj.uniqueId  in down):
                                    down.append(miscDataObj.uniqueId)
                                    update_trend(miscDataObj.uniqueId,up,down)

                           

                            if is_inside_roi(bbox_points, ROI) and  not (miscDataObj.uniqueId in general_roi):
                                
                                general_roi.append(miscDataObj.uniqueId)

                            
                            if miscDataObj.uniqueId in general_roi and miscDataFrame.age > 2:
                               
                                
                                if orientation == 'Up' and  not (miscDataObj.uniqueId in up_roi):
                                    up_roi.append(miscDataObj.uniqueId)
                                    update_trend(miscDataObj.uniqueId,up_roi,down_roi)

                                if orientation == 'Down' and  not (miscDataObj.uniqueId in down_roi):
                                    down_roi.append(miscDataObj.uniqueId)
                                    update_trend(miscDataObj.uniqueId,up_roi,down_roi)
                        
                
        try:
            l_user=l_user.next
        except StopIteration:
            break
    

    return Gst.PadProbeReturn.OK






def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if not platform_info.is_integrated_gpu() and name.find("nvv4l2decoder") != -1:
        # Use CUDA unified memory in the pipeline so frames can be easily accessed on CPU in Python.
        # 0: NVBUF_MEM_CUDA_DEVICE, 1: NVBUF_MEM_CUDA_PINNED, 2: NVBUF_MEM_CUDA_UNIFIED
        # Dont use direct macro here like NVBUF_MEM_CUDA_UNIFIED since nvv4l2decoder uses a
        # different enum internally
        Object.set_property("cudadec-memtype", 2)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)

def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(args):
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN] <folder to save frames>\n" % args[0])
        sys.exit(1)

    global perf_data
    perf_data = PERF_DATA(len(args) - 2)
    number_sources = len(args) - 2

    global folder_name
    folder_name = args[-1]
    if path.exists(folder_name):
        sys.stderr.write("The output folder %s already exists. Please remove it first.\n" % folder_name)
        sys.exit(1)

    os.mkdir(folder_name)
    print("Frames will be saved in ", folder_name)
    global platform_info
    platform_info = PlatformInfo()
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        os.mkdir(folder_name + "/stream_" + str(i))
        frame_count["stream_" + str(i)] = 0
        saved_count["stream_" + str(i)] = 0
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i + 1]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    #-------------------------------------------------------------
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")
    #-------------------------------------------------------------
    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.
    print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")

    

    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    if platform_info.is_integrated_gpu():
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        if platform_info.is_platform_aarch64():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        else:
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    #------------------------------------------------------------------------
    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('dstest_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
    #------------------------------------------------------------------------

    pgie.set_property('config-file-path', "dstest_imagedata_config.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if (pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ",
              number_sources, " \n")
        pgie.set_property("batch-size", number_sources)
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    sink.set_property("sync", 0)
    sink.set_property("qos", 0)

    if not platform_info.is_integrated_gpu():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        if platform_info.is_wsl():
            #opencv functions like cv2.line and cv2.putText is not able to access NVBUF_MEM_CUDA_UNIFIED memory
            #in WSL systems due to some reason and gives SEGFAULT. Use NVBUF_MEM_CUDA_PINNED memory for such
            #usecases in WSL. Here, nvvidconv1's buffer is used in tiler sink pad probe and cv2 operations are
            #done on that.
            print("using nvbuf_mem_cuda_pinned memory for nvvidconv1\n")
            vc_mem_type = int(pyds.NVBUF_MEM_CUDA_PINNED)
            nvvidconv1.set_property("nvbuf-memory-type", vc_mem_type)
        else:
            nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
        

        
        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    


    

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args[:-1]):
        if i != 0:
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    

    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

    generate_video()



if __name__ == '__main__':
    main(sys.argv)
    
    sys.exit()
    
    
