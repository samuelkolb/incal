import os


try:
    from telegram.ext import Updater
    from telegram import ParseMode
except ImportError:
    Updater = None
    ParseMode = None

from autodora.observe import ProgressObserver


class TelegramObserver(ProgressObserver):
    def __init__(self):
        super().__init__()
        if Updater is None:
            raise RuntimeError("The TelegramObserver requires additional packages, please install them"
                               "(e.g. pip install autodora[telegram]).")
        try:
            self.updater = Updater(os.environ["TELEGRAM_BOT_TOKEN"])
            self.chat_id = os.environ["TELEGRAM_CHAT_ID"]
        except KeyError:
            raise RuntimeError("Please provide environment variables TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        self.message_id = None
        self.total = None
        self.timed_out = 0
        self.done = 0
        self.errors = 0
        self.error_messages = []
        self.run_id = None
        self.run_platform = None

    def send_message(self, done=False):
        template = "*{run} ({start})*\n**{platform}**\n{done} succeeded," \
                   " {timed_out} timed out, {failed} failed{messages}"
        total_done = self.done + self.timed_out + self.errors
        message = template.format(
            start="{total_done} of {total}".format(total_done=total_done, total=self.total) if not done else "done",
            run=self.run_id,
            done=self.done,
            timed_out=self.timed_out,
            failed=self.errors,
            messages=("".join(["\n{}".format(m) for m in ["\n*Error messages:*"] + self.error_messages])
            if len(self.error_messages) > 0 else ""),
            platform=self.run_platform,
        )
        if self.message_id is None:
            self.message_id = self.updater.bot.send_message(chat_id=self.chat_id, text=message,
                                                            parse_mode=ParseMode.MARKDOWN).message_id
        else:
            print(message, type(message))
            self.updater.bot.edit_message_text(chat_id=self.chat_id, text=message, message_id=self.message_id,
                                               parse_mode=ParseMode.MARKDOWN)

    def run_started(self, platform, name, run_count, run_date, experiment_count):
        self.total = experiment_count
        self.run_id = "{}.{}".format(name, run_count)
        self.run_platform = platform
        self.send_message()

    def experiment_started(self, index, experiment):
        pass

    def experiment_finished(self, index, experiment):
        self.done += 1
        self.send_message()

    def experiment_interrupted(self, index, experiment):
        self.timed_out += 1
        self.send_message()

    def experiment_failed(self, index, experiment):
        self.errors += 1
        self.send_message()

    def run_finished(self, platform, name, run_count, run_date):
        self.send_message(True)
