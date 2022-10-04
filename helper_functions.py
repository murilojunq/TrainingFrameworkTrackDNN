from datetime import datetime

def get_timestamp():
    date_time = datetime.now()
    stamp = "{year:02}{month:02}{day:02}_{hour:02}{minute:02}{second:02}".format(year=date_time.year,
                                                               month=date_time.month,
                                                               day=date_time.day,
                                                               hour=date_time.hour,
                                                               minute=date_time.minute,
                                                               second=date_time.second)

    return stamp
