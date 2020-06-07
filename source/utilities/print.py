from source.utilities.config.hyperparameters import HyperParameters


def print_message(message: str,
                  to_file: bool = False,
                  config: HyperParameters = None,
                  mode: str = 'a',
                  ):
    """
    Prints the given message on the console or writes it to a log file corresponding to the hostname.
    :param message: the message
    :param to_file: True if the message shall be written to a file, False else. Defaults to False.
    :param config: the config containing the hostname
    :param mode: the mode used to write to the file. Defaults to a.
    """
    if to_file and config is None:
        raise Exception('A configuration must be provided when printing to a file, but config is None')
    if to_file:
        file_name = 'log_{}'.format(config.hostname)
        with open(file_name, mode) as f:
            f.write(message + '\n')
    else:
        print(message)

