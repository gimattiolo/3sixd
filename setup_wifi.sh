#!/usr/bin/bash

networkctl up wlan0
iw dev wlan0 scan | grep SSID
wpa_passphrase "cat&pear" "iosperiamochemelacavo" >> /etc/wpa_supplicant/wpa_supplicant-wlan0.conf
networkctl up wlan0
systemctl restart systemd-networkd.service
systemctl restart wpa_supplicant@wlan0.service

ifconfig wlan0

ping google.com
