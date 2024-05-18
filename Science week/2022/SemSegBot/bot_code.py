from aiogram import Dispatcher,Bot,types,executor
import cv2
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import main2

#import script_code
Token='5694933403:AAHPzNX7l_G_QniF3CZQ4NjfoasrdC7kN-E'
bot=Bot(token=Token)
dp = Dispatcher(bot)

model = tf.keras.models.load_model('efficientnetb1_384_augment', compile=False)
colormap = loadmat(
        "human_colormap.mat"
    )["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)

@dp.message_handler(commands='start')
async def start(message:types.Message):
    await message.answer('Отправьте изображение для обработки')

@dp.message_handler(content_types=['photo'])
async def image(message:types.Message):
    try:
        user_id=message.from_user.id
        path=f"photos/{str(user_id)}.jpg"
        await message.photo[-1].download(destination_file=path)
        main2.begin(user_id).processing(model=model,colormap=colormap)


        await bot.send_photo(message.chat.id,photo=open(f"photos/{str(user_id)}_processed.jpg",'rb'))
        await bot.send_photo(message.chat.id, photo=open(f"photos/{str(user_id)}_segmentation_mask.jpg", 'rb'))
    except Exception as ex:
        await message.answer('Что-то пошло не так, попробуйте еще раз загрузить изображение')
        print(ex)

if __name__ == '__main__':
    executor.start_polling(dp,skip_updates=True)
