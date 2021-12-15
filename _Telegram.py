# pragma pylint: disable=unused-argument, unused-variable, protected-access, invalid-name

"""
This module manage Telegram communication
"""
import json
import logging
import re
from datetime import date, datetime, timedelta
from html import escape
from itertools import chain
from math import isnan
from typing import Any, Callable, Dict, List, Optional, Union

import arrow
from tabulate import tabulate
from telegram import (CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton,
                      ParseMode, ReplyKeyboardMarkup, Update)
from telegram.error import BadRequest, NetworkError, TelegramError
from telegram.ext import CallbackContext, CallbackQueryHandler, CommandHandler, Updater
from telegram.utils.helpers import escape_markdown



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

        chat_id = int(self._config[self._config['user']]['telegram']['chat_id'])
        if cchat_id != chat_id:
            logger.info(
                'Rejected unauthorized message from: %s',
                update.message.chat_id
            )
            return wrapper

        logger.debug(
            'Executing handler: %s for chat_id: %s',
            command_handler.__name__,
            chat_id
        )
        try:
            return command_handler(self, *args, **kwargs)
        except BaseException:
            logger.exception('Exception occurred within Telegram module')

    return wrapper


class _Telegram():
    """  This class handles all telegram communication """

    def __init__(self, config: ConfigObj) -> None:
        """
        Init the Telegram call, and init the super class RPCHandler
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """

        self._updater: Updater
        self._init_keyboard()
        self._init()

    def _init_keyboard(self) -> None:
        """
        Validates the keyboard configuration from telegram config
        section.
        """
        self._keyboard: List[List[Union[str, KeyboardButton]]] = [
            ['/status'],
            ['/start', '/stop', '/help', '/version']
        ]
        # do not allow commands with mandatory arguments and critical cmds
        # like /forcesell and /forcebuy
        # TODO: DRY! - its not good to list all valid cmds here. But otherwise
        #       this needs refactoring of the whole telegram module (same
        #       problem in _help()).
        valid_keys: List[str] = [r'/start$', r'/stop$', r'/status$',
                                 r'/help$', r'/version$']
        # Create keys for generation
        valid_keys_print = [k.replace('$', '') for k in valid_keys]


    def _init(self) -> None:
        """
        Initializes this module with the given config,
        registers all known command handlers
        and starts polling for message updates
        """
        self._updater = Updater(token=self._config['users'][self._config['user']]['telegram']['token'], workers=0,
                                use_context=True)

        # Register command handler and start telegram message polling
        handles = [
            CommandHandler('status', self._status),
            CommandHandler('start', self._start),
            CommandHandler('stop', self._stop),
            CommandHandler('help', self._help),
            CommandHandler('version', self._version),
        ]
        callbacks = [
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
        logger.info(
            'rpc.telegram is listening for following commands: %s',
            [h.command for h in handles]
        )

    def cleanup(self) -> None:
        """
        Stops all running telegram threads.
        :return: None
        """
        self._updater.stop()

    def send_msg(self, msg: Dict[str, Any]) -> None:
        """ Send a message to telegram channel """
        default_noti = 'on'
        msg_type = msg['type']
        noti = ''

        noti = self._config['telegram'] \
            .get('notification_settings', {}).get(str(msg_type), default_noti)

        if noti == 'off':
            logger.info(f"Notification '{msg_type}' not sent.")
            # Notification disabled
            return

        if msg_type == RPCMessageType.BUY:
            message = self._format_buy_msg(msg)

        elif msg_type == RPCMessageType.STATUS:
            message = '*Status:* `{status}`'.format(**msg)

        elif msg_type == RPCMessageType.WARNING:
            message = '\N{WARNING SIGN} *Warning:* `{status}`'.format(**msg)

        elif msg_type == RPCMessageType.STARTUP:
            message = '{status}'.format(**msg)

        else:
            raise NotImplementedError('Unknown message type: {}'.format(msg_type))

        self._send_msg(message, disable_notification=(noti == 'silent'))

    def _get_sell_emoji(self, msg):
        """
        Get emoji for sell-side
        """

        if float(msg['profit_percent']) >= 5.0:
            return "\N{ROCKET}"
        elif float(msg['profit_percent']) >= 0.0:
            return "\N{EIGHT SPOKED ASTERISK}"
        elif msg['sell_reason'] == "stop_loss":
            return"\N{WARNING SIGN}"
        else:
            return "\N{CROSS MARK}"

    @authorized_only
    def _status(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /status.
        Returns the current TradeThread status
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        if context.args and 'table' in context.args:
            self._status_table(update, context)
            return

        try:

            # Check if there's at least one numerical ID provided.
            # If so, try to get only these trades.
            trade_ids = []
            if context.args and len(context.args) > 0:
                trade_ids = [int(i) for i in context.args if i.isnumeric()]

            results = self._rpc._rpc_trade_status(trade_ids=trade_ids)

            messages = []
            for r in results:
                r['open_date_hum'] = arrow.get(r['open_date']).humanize()
                lines = [
                    "*Trade ID:* `{trade_id}` `(since {open_date_hum})`",
                    "*Current Pair:* {pair}",
                    "*Amount:* `{amount} ({stake_amount} {base_currency})`",
                    "*Buy Tag:* `{buy_tag}`" if r['buy_tag'] else "",
                    "*Open Rate:* `{open_rate:.8f}`",
                    "*Close Rate:* `{close_rate}`" if r['close_rate'] else "",
                    "*Current Rate:* `{current_rate:.8f}`",
                    ("*Current Profit:* " if r['is_open'] else "*Close Profit: *")
                    + "`{profit_pct:.2f}%`",
                ]
                if (r['stop_loss_abs'] != r['initial_stop_loss_abs']
                        and r['initial_stop_loss_pct'] is not None):
                    # Adding initial stoploss only if it is different from stoploss
                    lines.append("*Initial Stoploss:* `{initial_stop_loss_abs:.8f}` "
                                 "`({initial_stop_loss_pct:.2f}%)`")

                # Adding stoploss and stoploss percentage only if it is not None
                lines.append("*Stoploss:* `{stop_loss_abs:.8f}` " +
                             ("`({stop_loss_pct:.2f}%)`" if r['stop_loss_pct'] else ""))
                lines.append("*Stoploss distance:* `{stoploss_current_dist:.8f}` "
                             "`({stoploss_current_dist_pct:.2f}%)`")
                if r['open_order']:
                    if r['sell_order_status']:
                        lines.append("*Open Order:* `{open_order}` - `{sell_order_status}`")
                    else:
                        lines.append("*Open Order:* `{open_order}`")

                # Filter empty lines using list-comprehension
                messages.append("\n".join([line for line in lines if line]).format(**r))

            for msg in messages:
                self._send_msg(msg)

        except RPCException as e:
            self._send_msg(str(e))


    @authorized_only
    def _start(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /start.
        Starts TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_start()
        self._send_msg('Status: `{status}`'.format(**msg))

    @authorized_only
    def _stop(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /stop.
        Stops TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_stop()
        self._send_msg('Status: `{status}`'.format(**msg))


    @staticmethod
    def _layout_inline_keyboard(buttons: List[InlineKeyboardButton],
                                cols=3) -> List[List[InlineKeyboardButton]]:
        return [buttons[i:i + cols] for i in range(0, len(buttons), cols)]



    @authorized_only
    def _help(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /help.
        Show commands of the bot
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        forcebuy_text = ("*/forcebuy <pair> [<rate>]:* `Instantly buys the given pair. "
                         "Optionally takes a rate at which to buy.` \n")
        message = ("*/start:* `Starts the trader`\n"
                   "*/stop:* `Stops the trader`\n"
                   "*/status <trade_id>|[table]:* `Lists all open trades`\n"
                   "         *<trade_id> :* `Lists one or more specific trades.`\n"
                   "                        `Separate multiple <trade_id> with a blank space.`\n"
                   "         *table :* `will display trades in a table`\n"
                   "                `pending buy orders are marked with an asterisk (*)`\n"
                   "                `pending sell orders are marked with a double asterisk (**)`\n"
                   "*/trades [limit]:* `Lists last closed trades (limited to 10 by default)`\n"
                   "*/profit [<n>]:* `Lists cumulative profit from all finished trades, "
                   "over the last n days`\n"
                   "*/forcesell <trade_id>|all:* `Instantly sells the given trade or all trades, "
                   "regardless of profit`\n"
                   f"{forcebuy_text if self._config.get('forcebuy_enable', False) else ''}"
                   "*/delete <trade_id>:* `Instantly delete the given trade in the database`\n"
                   "*/performance:* `Show performance of each finished trade grouped by pair`\n"
                   "*/daily <n>:* `Shows profit or loss per day, over the last n days`\n"
                   "*/stats:* `Shows Wins / losses by Sell reason as well as "
                   "Avg. holding durationsfor buys and sells.`\n"
                   "*/count:* `Show number of active trades compared to allowed number of trades`\n"
                   "*/locks:* `Show currently locked pairs`\n"
                   "*/unlock <pair|id>:* `Unlock this Pair (or this lock id if it's numeric)`\n"
                   "*/balance:* `Show account balance per currency`\n"
                   "*/stopbuy:* `Stops buying, but handles open trades gracefully` \n"
                   "*/reload_config:* `Reload configuration file` \n"
                   "*/show_config:* `Show running configuration` \n"
                   "*/logs [limit]:* `Show latest logs - defaults to 10` \n"
                   "*/whitelist:* `Show current whitelist` \n"
                   "*/blacklist [pair]:* `Show current blacklist, or adds one or more pairs "
                   "to the blacklist.` \n"
                   "*/edge:* `Shows validated pairs by Edge if it is enabled` \n"
                   "*/help:* `This help message`\n"
                   "*/version:* `Show version`")

        self._send_msg(message)

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
        if reload_able and self._config['telegram'].get('reload', True):
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
                    self._config['telegram']['chat_id'],
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
                    self._config['telegram']['chat_id'],
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
