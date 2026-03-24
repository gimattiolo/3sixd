media-ctl --reset

#v4l2-ctl --device /dev/video0 --set-fmt-video=640,height=480,pixelformat=MJPG --stream-mmap --stream-to=./output.jpg --stream-count=1

###Run the following command to see if you have the camera(s) detected
v4l2-ctl --list-devices

v4l2-ctl --list-formats-ext

#exit

FMT=UYVY8_1X16/640x480

echo $FMT

###Setup links:
###mipi_csi0:
media-ctl -l "'ov5640 2-003c':0->'csidev-4ad30000.csi':0 [1]"
media-ctl -l "'csidev-4ad30000.csi':1 -> '4ac10000.syscon:formatter@20':0 [1]"

###Setup the media pipeline:
media-ctl -V "'ov5640 2-003c':0 [fmt: $FMT field:none]"
media-ctl -V "'csidev-4ad30000.csi':0 [fmt: $FMT  field:none]"
media-ctl -V "'4ac10000.syscon:formatter@20':0 [fmt: $FMT field:none]"
media-ctl -V "'crossbar':2 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.0':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.1':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.2':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.3':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.4':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.5':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.6':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.7':0 [fmt: $FMT field:none]"

###confirmation pipeline config was succesful
#media-ctl --get-v4l2 "'ov5640 2-003c':0"

###preview the camera on the display for different resolutions:
###VGA 640x480 30fps:
###mipi_csi0:
media-ctl -V "'ov5640 2-003c':0 [fmt: $FMT field:none]"
media-ctl -V "'csidev-4ad30000.csi':0 [fmt: $FMT field:none]"
media-ctl -V "'4ac10000.syscon:formatter@20':0 [fmt: $FMT field:none]"
media-ctl -V "'crossbar':2 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.0':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.1':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.2':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.3':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.4':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.5':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.6':0 [fmt: $FMT field:none]"
media-ctl -V "'mxc_isi.7':0 [fmt: $FMT field:none]"
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,format=YUY2 ! autovideosink
