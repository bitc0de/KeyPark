from telegram.ext import Updater, CommandHandler
from parking import *



def pk(bot, update):
    chat_id = bot.get_updates()[-1].message.chat_id
    update.message.reply_text("Cargando...")
    captura()
    clase().pk()
    var1=total()
    bot.send_photo(chat_id=chat_id, photo=open('captura.png', 'rb'))
    print ("libres: " + str(var1[0]))
    print("ocupados: " + str(var1[1]))
    update.message.reply_text("Libres: " + str(var1[0]) + "\nOcupados: " + str(var1[1]))




#total()

updater = Updater('851520479:AAGa7fu6EHoy6mFfiBow2AdjOp5G6IB79ZU')


updater.dispatcher.add_handler(CommandHandler('start', pk))

updater.start_polling()
updater.idle()