from typing import Any, Callable, Dict, List, Optional, Union
from configobj import ConfigObj
from telegram import (CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton,
                      ParseMode, ReplyKeyboardMarkup, Update)
from telegram.error import BadRequest, NetworkError, TelegramError
from telegram.ext import CallbackContext, CallbackQueryHandler, CommandHandler, Updater
from telegram.utils.helpers import escape_markdown
import logging
from __init__ import __version__


logger = logging.getLogger(__name__)
logger.debug('Included module rpc.telegram ...')

MAX_TELEGRAM_MESSAGE_LENGTH = 4096


def authorized_only(command_handler: Callable[..., None]) -> Callable[..., Any]:
    """
    Decorator to check if the message comes from the correct chat_id
    :param command_handler: Telegram CommandHandler
    :return: decorated function
    """

    def wrapper(self, *args, **kwargs):
        """ Decorator logic """
        update = kwargs.get('update') or args[0]

        # Reject unauthorized messages
        if update.callback_query:
            cchat_id = int(update.callback_query.message.chat.id)
        else:
            cchat_id = int(update.message.chat_id)

        chat_id = int(self._config['users'][self._config['user']]['telegram']['chat_id'])
        if cchat_id != chat_id:
            return wrapper
        try:
            return command_handler(self, *args, **kwargs)
        except BaseException:
            logger.exception('Exception occurred within Telegram module')

    return wrapper

class Telegram:
    def __init__(self, config: ConfigObj) -> None:
        """
        Init the Telegram call, and init the super class RPCHandler
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """

        self._config = config
        self._updater: Updater
        self._init_keyboard()
        self.init(config)

    def init(self, config):
        if config['users'][config['user']].get('telegram') != None:
            self.enabled = config['users'][config['user']]['telegram'].as_bool('enabled')
            self.token = config['users'][config['user']]['telegram']['token']
            self.chat_id = config['users'][config['user']]['telegram']['chat_id']


        """
        Initializes this module with the given config,
        registers all known command handlers
        and starts polling for message updates
        """
        self._updater = Updater(token=config['users'][config['user']]['telegram']['token'], workers=0,
                                use_context=True)

        # Register command handler and start telegram message polling
        handles = [
            CommandHandler('status', self._status),
            CommandHandler('balance', self._balance),
            CommandHandler('start', self._start),
            CommandHandler('stop', self._stop),
            CommandHandler('help', self._help),
            CommandHandler('version', self._version),
        ]
        callbacks = [
            CallbackQueryHandler(self._balance, pattern='update_balance'),
        ]
        for handle in handles:
            self._updater.dispatcher.add_handler(handle)

        for callback in callbacks:
            self._updater.dispatcher.add_handler(callback)

        self._updater.start_polling(
            bootstrap_retries=-1,
            timeout=30,
            read_latency=60,
            drop_pending_updates=True,
        )

    def _init_keyboard(self) -> None:
        """
        Validates the keyboard configuration from telegram config
        section.
        """
        self._keyboard: List[List[Union[str, KeyboardButton]]] = [
            [ '/start', '/stop', '/help', '/version']
        ]
        # do not allow commands with mandatory arguments and critical cmds
        # like /forcesell and /forcebuy
        # TODO: DRY! - its not good to list all valid cmds here. But otherwise
        #       this needs refactoring of the whole telegram module (same
        #       problem in _help()).
        valid_keys: List[str] = [r'/start$', r'/stop$', r'/help$', r'/version$']
        # Create keys for generation
        valid_keys_print = [k.replace('$', '') for k in valid_keys]

        # custom keyboard specified in config.json
        cust_keyboard = self._config['users'][config['user']]['telegram'].get('keyboard', [])
        if cust_keyboard:
            combined = "(" + ")|(".join(valid_keys) + ")"
            # check for valid shortcuts
            invalid_keys = [b for b in chain.from_iterable(cust_keyboard)
                            if not re.match(combined, b)]
            if len(invalid_keys):
                err_msg = ('config.telegram.keyboard: Invalid commands for '
                           f'custom Telegram keyboard: {invalid_keys}'
                           f'\nvalid commands are: {valid_keys_print}')
                raise OperationalException(err_msg)
            else:
                self._keyboard = cust_keyboard
                logger.info('using custom keyboard from '
                            f'config.json: {self._keyboard}')

    @authorized_only
    def _status(self):
        console.log('status')
    def _start(self):
        console.log('start')
    def _stop(self):
        console.log('stop')
    def _help(self):
        console.log('help')
    def _balance(self):
        console.log('balance')


    def _init_keyboard(self) -> None:
        """
        Validates the keyboard configuration from telegram config
        section.
        """
        self._keyboard: List[List[Union[str, KeyboardButton]]] = [
            ['/balance'],
            ['/status'],
            ['/start', '/stop', '/help']
        ]
        # do not allow commands with mandatory arguments and critical cmds
        # like /forcesell and /forcebuy
        # TODO: DRY! - its not good to list all valid cmds here. But otherwise
        #       this needs refactoring of the whole telegram module (same
        #       problem in _help()).
        valid_keys: List[str] = [r'/start$', r'/stop$', r'/status$',
                                 r'/stats$', r'/balance$',
                                 r'/help$', r'/version$']
        # Create keys for generation
        valid_keys_print = [k.replace('$', '') for k in valid_keys]

        # custom keyboard specified in config.json
        cust_keyboard = self._config['users'][self._config['user']]['telegram'].get('keyboard', [])
        if cust_keyboard:
            combined = "(" + ")|(".join(valid_keys) + ")"
            # check for valid shortcuts
            invalid_keys = [b for b in chain.from_iterable(cust_keyboard)
                            if not re.match(combined, b)]
            if len(invalid_keys):
                err_msg = ('config.telegram.keyboard: Invalid commands for '
                           f'custom Telegram keyboard: {invalid_keys}'
                           f'\nvalid commands are: {valid_keys_print}')
                raise OperationalException(err_msg)
            else:
                self._keyboard = cust_keyboard
                logger.info('using custom keyboard from '
                            f'config.json: {self._keyboard}')


    @authorized_only
    def _version(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /version.
        Show version information
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        self._send_msg('*Version:* `{}`'.format(__version__))

    def cleanup(self) -> None:
        """
        Stops all running telegram threads.
        :return: None
        """
        self._updater.stop()

    def _update_msg(self, query: CallbackQuery, msg: str, callback_path: str = "",
                    reload_able: bool = False, parse_mode: str = ParseMode.MARKDOWN) -> None:
        if reload_able:
            reply_markup = InlineKeyboardMarkup([
                [InlineKeyboardButton("Refresh", callback_data=callback_path)],
            ])
        else:
            reply_markup = InlineKeyboardMarkup([[]])
        msg += "\nUpdated: {}".format(datetime.now().ctime())
        if not query.message:
            return
        chat_id = query.message.chat_id
        message_id = query.message.message_id

        try:
            self._updater.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=msg,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
        except BadRequest as e:
            if 'not modified' in e.message.lower():
                pass
            else:
                logger.warning('TelegramError: %s', e.message)
        except TelegramError as telegram_err:
            logger.warning('TelegramError: %s! Giving up on that message.', telegram_err.message)

    def _send_msg(self, msg: str, parse_mode: str = ParseMode.MARKDOWN,
                  disable_notification: bool = False,
                  keyboard: List[List[InlineKeyboardButton]] = None,
                  callback_path: str = "",
                  reload_able: bool = False,
                  query: Optional[CallbackQuery] = None) -> None:
        """
        Send given markdown message
        :param msg: message
        :param bot: alternative bot
        :param parse_mode: telegram parse mode
        :return: None
        """
        reply_markup: Union[InlineKeyboardMarkup, ReplyKeyboardMarkup]
        if query:
            self._update_msg(query=query, msg=msg, parse_mode=parse_mode,
                             callback_path=callback_path, reload_able=reload_able)
            return
        if reload_able and self._config['users'][self._config['user']]['telegram'].get('reload', True):
            reply_markup = InlineKeyboardMarkup([
                [InlineKeyboardButton("Refresh", callback_data=callback_path)]])
        else:
            if keyboard is not None:
                reply_markup = InlineKeyboardMarkup(keyboard, resize_keyboard=True)
            else:
                reply_markup = ReplyKeyboardMarkup(self._keyboard, resize_keyboard=True)
        try:
            try:
                self._updater.bot.send_message(
                    self._config['users'][self._config['user']]['telegram']['chat_id'],
                    text=msg,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    disable_notification=disable_notification,
                )
            except NetworkError as network_err:
                # Sometimes the telegram server resets the current connection,
                # if this is the case we send the message again.
                logger.warning(
                    'Telegram NetworkError: %s! Trying one more time.',
                    network_err.message
                )
                self._updater.bot.send_message(
                    self._config['users'][self._config['user']]['telegram']['chat_id'],
                    text=msg,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    disable_notification=disable_notification,
                )
        except TelegramError as telegram_err:
            logger.warning(
                'TelegramError: %s! Giving up on that message.',
                telegram_err.message
            )

