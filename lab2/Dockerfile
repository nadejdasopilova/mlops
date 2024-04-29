# Используем официальный образ Jenkins
FROM jenkins/jenkins:latest
# Переключаемся на пользователя root для установки пакетов
USER root
# Обновление списка пакетов и установка Python и pip
RUN apt-get update && \
apt-get install -y python3 python3-pip python3-venv
# Создание виртуального окружения
RUN python3 -m venv /opt/venv
# Активация виртуального окружения и установка пакетов
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
pip install -r "requirements.txt"
# Возвращаем пользователя Jenkins
USER jenkins