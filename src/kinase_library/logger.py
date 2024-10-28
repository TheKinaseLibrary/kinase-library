import logging
import io
import sys
from tqdm import tqdm

logger = logging.getLogger("kedro")


class TqdmToLoggerWithStatus(io.StringIO):
    """
    Output stream for TQDM which will output to logger module and also to StdOut.
    Uses tqdm's status_printer to manage in-place updating of the progress.
    """

    def __init__(self, logger, level=None):
        super(TqdmToLoggerWithStatus, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
        self.buf = ''
        self.status_printer = tqdm.status_printer(sys.stdout)

    def write(self, buf):
        # Update the buffer and print status
        self.buf = buf
        self.status_printer(buf)
        # Also print to stdout
        sys.stdout.write(buf)
        sys.stdout.flush()

    def flush(self):
        # Log the buffered message
        if self.buf.strip():
            self.logger.log(self.level, self.buf.strip())
            self.buf = ''  # Clear the buffer after logging
