def beat_gif_callback(beats, output=None):
    """Queue gif frames."""
    if len(beats) > 0:
        current_beat_time = time.perf_counter()
        duration = current_beat_time - beat_gif_callback.last_beat_time
        global root
        # cancel all scheduled gif frames
        for after_id in beat_gif_callback.after_ids:
            root.after_cancel(after_id)
        beat_gif_callback.after_ids.clear()
        # queue new frames
        global n_frames
        last_frame_time = 0
        for i in range(n_frames):
            frame_time = i * duration / n_frames
            if frame_time - last_frame_time < 1 / 120:
                continue
            beat_gif_callback.after_ids.append(root.after(int(frame_time * 1000), update, (i + n_frames // 2) % n_frames))
            last_frame_time = frame_time
        print(beats)
        beat_gif_callback.last_beat_time = current_beat_time
        if KEYBOARD_BACKLIGHTING:
            try:
                if beat_gif_callback.tick == 0:
                    win32api.PostMessage(keyboard_backlight_hwnd, win32con.WM_KEYDOWN, win32con.VK_END, 0x104f0001)
                    win32api.PostMessage(keyboard_backlight_hwnd, win32con.WM_KEYUP, win32con.VK_END, 0xd04f0001)
                elif beat_gif_callback.tick == 1:
                    win32api.PostMessage(keyboard_backlight_hwnd, win32con.WM_KEYDOWN, win32con.VK_LEFT, 0x104b0001)
                    win32api.PostMessage(keyboard_backlight_hwnd, win32con.WM_KEYUP, win32con.VK_LEFT, 0xd04b0001)
                elif beat_gif_callback.tick == 2:
                    win32api.PostMessage(keyboard_backlight_hwnd, win32con.WM_KEYDOWN, win32con.VK_LEFT, 0x104b0001)
                    win32api.PostMessage(keyboard_backlight_hwnd, win32con.WM_KEYUP, win32con.VK_LEFT, 0xd04b0001)
            except pywintypes.error as e:
                # this callback may act during the exit handler that is closing mblctr
                print(e)
            if beat_gif_callback.tick == 3:
                beat_gif_callback.tick = 0
            else:
                beat_gif_callback.tick += 1

def update(index):
    global label, frames
    """Change to the <index>th frame of gif."""
    label.configure(image=frames[index])

def toggle_show_window(_event=None):
    """Toggle the gif window between shown and hidden."""
    toggle_show_window.show = not toggle_show_window.show
    root.overrideredirect(toggle_show_window.show)
    root.wm_attributes('-topmost', toggle_show_window.show)

def load_gif(frames=[]):
    """Load gif frames from file."""
    # frames = []
    n_frames = 0
    while True:
        try:
            frames.append(PhotoImage(file='kirby.gif', format=f'gif -index {n_frames}'))
            n_frames += 1
            print(n_frames)
        except TclError:
            # finished all frames
            break
    return n_frames

def get_hwnd_if_keyboard_backlighting_callback(hwnd, keyboard_backlight_hwnd):
    """Check if the given window is the keyboard backlighting trackbar control. If it is, store the handle, and return false to stop enumeration."""
    if win32gui.GetWindowText(win32gui.GetWindow(hwnd, win32con.GW_CHILD)) == 'Keyboard Backlighting':
        keyboard_backlight_hwnd.append(win32gui.FindWindowEx(hwnd, 0, 'msctls_trackbar32', None))
        return False

def close_window_if_pid_to_close(hwnd, extra):
    """Try to close a window if it belongs to a pid that should be closed, and terminate the process explicitly if needed."""
    try:
        # get tid and pid of given window
        tid, pid = win32process.GetWindowThreadProcessId(hwnd)
        # check if pid should be closed
        if pid in pids_to_close:
            # send WM_CLOSE to window
            win32api.SendMessage(hwnd, win32con.WM_CLOSE, None, None)
    except pywintypes.error:
        # window may close unexpectedly and become invalid
        pass

def close_mblctr_then_exit(sig=None, frame=None):
    """Reset keyboard backlighting, close mblctr and related processes, and then exit."""
    process_stream.close = True
    # reset keyboard backlighting and wait for response
    win32api.PostMessage(keyboard_backlight_hwnd, win32con.WM_KEYDOWN, win32con.VK_HOME, 0x10470001)
    win32api.PostMessage(keyboard_backlight_hwnd, win32con.WM_KEYUP, win32con.VK_HOME, 0xd0470001)
    time.sleep(1 / 60)
    # close mblctr job handle, which should close mblctr and its child processes
    win32api.CloseHandle(mblctr_job)
    # get pids of processes to close
    for pid in win32process.EnumProcesses():
        try:
            hproc = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
            hmod = win32process.EnumProcessModulesEx(hproc, win32process.LIST_MODULES_ALL)[0]
            proc_name = win32process.GetModuleFileNameEx(hproc, hmod)
            if proc_name in names_of_procs_to_close:
                pids_to_close.add(pid)
            win32api.CloseHandle(hproc)
        except pywintypes.error:
            # processes may not grant rights requested or may become invalid unexpectedly
            pass
    # close windows of related processes
    win32gui.EnumWindows(close_window_if_pid_to_close, None)
    # try to terminate any remaining or new processes to close
    for pid in win32process.EnumProcesses():
        try:
            hproc = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ | win32con.PROCESS_TERMINATE, False, pid)
            hmod = win32process.EnumProcessModulesEx(hproc, win32process.LIST_MODULES_64BIT)[0]
            proc_name = win32process.GetModuleFileNameEx(hproc, hmod)
            if proc_name in names_of_procs_to_close:
                win32api.TerminateProcess(hproc, 0)
            win32api.CloseHandle(hproc)
        except pywintypes.error:
            # processes may not grant rights requested or may become invalid unexpectedly
            pass
    print('waiting join')
    processor_thread.join(3)
    sys.exit(0)

def on_close():
    process_stream.close = True
    print('waiting join')
    processor_thread.join(3)
    sys.exit(0)


if __name__ == '__main__':
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.processors import IOProcessor, process_online
    from madmom_loopback import process_stream

    import time

    import signal
    import sys

    from threading import Thread
    from tkinter import Tk, Label, PhotoImage, TclError

    KEYBOARD_BACKLIGHTING = False

    # initialize beat tracker
    in_processor = RNNBeatProcessor(fps=100, num_frames=1, origin='stream', online=True)
    beat_processor = DBNBeatTrackingProcessor(fps=100, online=True, verbose=True)
    processor = IOProcessor(in_processor, (beat_processor, beat_gif_callback))
    if KEYBOARD_BACKLIGHTING:
        import win32gui
        import win32con
        import win32api
        import win32process
        import win32event
        import win32job
        import pywintypes
        # prepare startup info for mblctr to be hidden
        py_startupinfo = win32process.STARTUPINFO()
        py_startupinfo.dwFlags += win32con.STARTF_USESHOWWINDOW
        py_startupinfo.wShowWindow += win32con.SW_HIDE
        # prepare SIGINT handler to close mblctr and other some other processes
        names_of_procs_to_close = {'C:\WINDOWS\System32\mobsync.exe', 'c:\Program Files\Dell\QuickSet\MobilityCenter.exe'}
        pids_to_close = set()
        signal.signal(signal.SIGINT, close_mblctr_then_exit)
        # create job and add a limit to kill all processes spawned under it when it is closed
        mblctr_job = win32job.CreateJobObject(None,    # jobAttributes
                                            ''       # name
        )
        limits = win32job.QueryInformationJobObject(mblctr_job, win32job.JobObjectExtendedLimitInformation)
        limits['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(mblctr_job, win32job.JobObjectExtendedLimitInformation, limits)
        # open mblctr suspended so that it doesn't spawn any child processes yet and assign it to the job created
        proc, thread, proc_id, thread_id = win32process.CreateProcess('C:\Windows\System32\mblctr.exe',     # appName
                                                                    'mblctr.exe',                         # commandLine
                                                                    None,                                 # processAttributes
                                                                    None,                                 # threadAttributes
                                                                    0,                                    # bInheritHandles
                                                                    win32con.CREATE_SUSPENDED,            # dwCreationFlags
                                                                    None,                                 # newEnvironment
                                                                    None,                                 # currentDirectory
                                                                    py_startupinfo                        # startupinfo
        )
        win32job.AssignProcessToJobObject(mblctr_job, proc)
        # store current foreground window and resume the thread of the new process
        current_window = win32gui.GetForegroundWindow()
        win32process.ResumeThread(thread)
        # wait for mblctr to become idle, then find its window
        if win32event.WaitForInputIdle(proc, 3000) != 0:
            print('mblctr took too long to become idle')
            sys.exit(0)
        proc_hwnd = win32gui.FindWindow(None, 'Windows Mobility Centre')
        # find keyboard backlight trackbar handle
        keyboard_backlight_hwnd = []
        while not keyboard_backlight_hwnd:
            time.sleep(1 / 60)
            try:
                win32gui.EnumChildWindows(proc_hwnd, get_hwnd_if_keyboard_backlighting_callback, keyboard_backlight_hwnd)
            except pywintypes.error:
                pass
        keyboard_backlight_hwnd = keyboard_backlight_hwnd[0]
        # reset foreground window stored earlier
        win32gui.SetForegroundWindow(current_window)
        # set focus to the keyboard backlight trackbar control
        win32process.AttachThreadInput(win32api.GetCurrentThreadId(), win32process.GetWindowThreadProcessId(proc_hwnd)[0], 1)
        win32gui.SetFocus(keyboard_backlight_hwnd)
    # beat track
    beat_gif_callback.tick = 0
    beat_gif_callback.after_ids = []
    beat_gif_callback.last_beat_time = time.perf_counter()
    process_stream.close = False
    # process_stream(processor)
    processor_thread = Thread(target=process_stream, args=(processor,))
    processor_thread.start()

    root = Tk()
    label = Label(root)
    label.pack()

    frames = []
    n_frames = load_gif(frames)
    label.configure(image=frames[0])

    # root.geometry(f'+{WINFO_X}+{WINFO_Y}')
    root.geometry(f'+{1725}+{786}')
    toggle_show_window.show = False
    toggle_show_window()
    root.bind('<Button-1>', toggle_show_window)
    root.wm_attributes('-transparentcolor', root.cget('bg'))
    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()
