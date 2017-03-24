#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:54:33 2017

@author: arvardaz
"""

import dropbox
import os

file_path = 'src/weights/vgg16_weights.npz'
dest_path = '/weights'

app_key = '77mgvmmod9jgpja'
access_token = '6Zj9MS9mIyAAAAAAAAAB7S_A4KNGrzrtqvn8EmeSYN9Y2YW1r8J1yA9RuNKsh-jU'

f = open(file_path)
file_size = os.path.getsize(file_path)

CHUNK_SIZE = 4 * 1024 * 1024

if file_size <= CHUNK_SIZE:

    print dbx.files_upload(f.read(), dest_path)

else:

    upload_session_start_result = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
    cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                               offset=f.tell())
    commit = dropbox.files.CommitInfo(path=dest_path)

    while f.tell() < file_size:
        if ((file_size - f.tell()) <= CHUNK_SIZE):
            print dbx.files_upload_session_finish(f.read(CHUNK_SIZE),
                                            cursor,
                                            commit)
        else:
            dbx.files_upload_session_append(f.read(CHUNK_SIZE),
                                            cursor.session_id,
                                            cursor.offset)
            cursor.offset = f.tell()

f.close()