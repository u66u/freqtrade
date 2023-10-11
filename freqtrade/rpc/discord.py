import logging

from freqtrade.constants import Config
from freqtrade.enums import RPCMessageType
from freqtrade.rpc import RPC
from freqtrade.rpc.webhook import Webhook


logger = logging.getLogger(__name__)


class Discord(Webhook):
    def __init__(self, rpc: 'RPC', config: Config):
        self._config = config
        self.rpc = rpc
        self.strategy = config.get('strategy', '')
        self.timeframe = config.get('timeframe', '')

        self._url = config['discord']['webhook_url']
        self._format = 'json'
        self._retries = 1
        self._retry_delay = 0.1
        self._timeout = self._config['discord'].get('timeout', 10)

    def cleanup(self) -> None:
        """
        Cleanup pending module resources.
        This will do nothing for webhooks, they will simply not be called anymore
        """
        pass

    def send_msg(self, msg) -> None:
        send_message = False
        if (msg['type'].value == "strategy_msg"):
            logger.info(f"Sending discord strategy message: {msg['msg']}")

            msg['strategy'] = self.strategy
            msg['timeframe'] = self.timeframe
            msg['exchange'] = self._config['exchange']['name']
            # fields = self.config['discord'].get(msg['type'].value)
            fields = self._config['discord'].get('rows_strategy_msg')
            color = 0xFF6600

            title = msg['type'].value
            
            send_message = True

        elif (msg['type'].value in ["status", "warning"]):
            # logger.info(f"Sending discord strategy message: {msg['msg']}")

            msg['strategy'] = self.strategy
            msg['timeframe'] = self.timeframe
            msg['exchange'] = self._config['exchange']['name']
            # fields = self.config['discord'].get(msg['type'].value)
            fields = self._config['discord'].get('rows_status')
            
            color = 0x008000
            if (msg['status'] != 'running'):
                color = 0xFF0000
            title = msg['type'].value
            
            send_message = True

        elif (msg['type'].value in ["entry_cancel", "exit_cancel"]):
            # logger.info(f"Sending discord strategy message: {msg['msg']}")

            msg['strategy'] = self.strategy
            msg['timeframe'] = self.timeframe
            # msg['exchange'] = self._config['exchange']['name']
            # fields = self.config['discord'].get(msg['type'].value)
            fields = self._config['discord'].get(msg['type'].value)
            
            color = 0xFF0000
            # title = msg['type'].value
            title = f"Trade #{msg['trade_id']}: {msg['pair']} {msg['type'].value}"
            
            send_message = True

        elif ((msg['type'].value in self._config['discord'])
              and (('enabled' not in self._config['discord'][msg['type'].value])
                   or (self._config['discord'][msg['type'].value]['enabled'] is True))):
            logger.info(f"Sending discord message: {msg}")

            msg['strategy'] = self.strategy
            msg['timeframe'] = self.timeframe
            # fields = self.config['discord'].get(msg['type'].value)
            fields = self._config['discord'][msg['type'].value].get('rows')
            if msg['sub_trade']:
                fields = self._config['discord'][msg['type'].value].get('rows_sub_trade')
            
            color = 0x0000FF
            if ((msg['type'] in (RPCMessageType.ENTRY, RPCMessageType.ENTRY_FILL))
                and msg['sub_trade']):
                color = 0x00FFFF

            if msg['type'] in (RPCMessageType.EXIT, RPCMessageType.EXIT_FILL):
                profit_ratio = msg.get('profit_ratio')
                color = ((0x00FF00 if profit_ratio > 0 else 0xFF00FF) if msg['sub_trade']
                         else (0x008000 if profit_ratio > 0 else 0xFF0000))

            title = msg['type'].value
            if 'pair' in msg:
                title = f"Trade #{msg['trade_id']}: {msg['pair']} {msg['type'].value}"
            if ('pair' in msg) and msg['sub_trade']:
                title = f"Trade #{msg['trade_id']}: {msg['pair']} sub_{msg['type'].value}"
            
            send_message = True

        if send_message:
            embeds = [{
                'title': title,
                'color': color,
                'fields': [],

            }]
            for f in fields:
                for k, v in f.items():
                    if (k == "Leverage"):
                        if msg.get('leverage', 1.0) == 1.0:
                            continue
                    v = v.format(**msg)
                    embeds[0]['fields'].append(
                        {'name': k, 'value': v, 'inline': True})

            # Send the message to discord channel
            payload = {'embeds': embeds}
            self._send_msg(payload)
