(function (global) {
    'use strict';

    const PI = Math.PI;
    const A = 6378245.0;
    const EE = 0.006693421622965943;

    function outsideChina(lng, lat) {
        return lng < 72.004 || lng > 137.8347 || lat < 0.8293 || lat > 55.8271;
    }

    function transformLat(lng, lat) {
        let value = -100 + 2 * lng + 3 * lat + 0.2 * lat * lat +
            0.1 * lng * lat + 0.2 * Math.sqrt(Math.abs(lng));
        value += (20 * Math.sin(6 * lng * PI) + 20 * Math.sin(2 * lng * PI)) * 2 / 3;
        value += (20 * Math.sin(lat * PI) + 40 * Math.sin(lat / 3 * PI)) * 2 / 3;
        value += (160 * Math.sin(lat / 12 * PI) + 320 * Math.sin(lat * PI / 30)) * 2 / 3;
        return value;
    }

    function transformLng(lng, lat) {
        let value = 300 + lng + 2 * lat + 0.1 * lng * lng +
            0.1 * lng * lat + 0.1 * Math.sqrt(Math.abs(lng));
        value += (20 * Math.sin(6 * lng * PI) + 20 * Math.sin(2 * lng * PI)) * 2 / 3;
        value += (20 * Math.sin(lng * PI) + 40 * Math.sin(lng / 3 * PI)) * 2 / 3;
        value += (150 * Math.sin(lng / 12 * PI) + 300 * Math.sin(lng / 30 * PI)) * 2 / 3;
        return value;
    }

    function toAmapCoordinate(lng, lat) {
        lng = Number(lng);
        lat = Number(lat);
        if (outsideChina(lng, lat)) return {lng: lng, lat: lat};

        let dLat = transformLat(lng - 105, lat - 35);
        let dLng = transformLng(lng - 105, lat - 35);
        const radLat = lat / 180 * PI;
        let magic = Math.sin(radLat);
        magic = 1 - EE * magic * magic;
        const sqrtMagic = Math.sqrt(magic);
        dLat = dLat * 180 / ((A * (1 - EE)) / (magic * sqrtMagic) * PI);
        dLng = dLng * 180 / (A / sqrtMagic * Math.cos(radLat) * PI);
        return {lng: lng + dLng, lat: lat + dLat};
    }

    function getAccuratePosition(options) {
        options = options || {};
        const targetAccuracy = options.targetAccuracy || 50;
        const maxWait = options.maxWait || 12000;

        return new Promise(function (resolve, reject) {
            if (!navigator.geolocation) {
                reject(new Error('浏览器不支持GPS定位'));
                return;
            }

            let best = null;
            let watchId = null;
            let timer = null;
            let settled = false;

            function cleanup() {
                if (watchId !== null) navigator.geolocation.clearWatch(watchId);
                if (timer !== null) clearTimeout(timer);
            }

            function finish(position) {
                if (settled) return;
                settled = true;
                cleanup();
                const mapCoordinate = toAmapCoordinate(position.coords.longitude, position.coords.latitude);
                resolve({
                    lng: mapCoordinate.lng,
                    lat: mapCoordinate.lat,
                    accuracy: Number(position.coords.accuracy) || 0,
                    rawLng: position.coords.longitude,
                    rawLat: position.coords.latitude
                });
            }

            function fail(error) {
                if (settled) return;
                settled = true;
                cleanup();
                reject(error);
            }

            watchId = navigator.geolocation.watchPosition(function (position) {
                if (!best || position.coords.accuracy < best.coords.accuracy) best = position;
                if (position.coords.accuracy <= targetAccuracy) finish(position);
            }, function (error) {
                if (best) finish(best);
                else fail(error);
            }, {
                enableHighAccuracy: true,
                maximumAge: 0,
                timeout: maxWait
            });

            timer = setTimeout(function () {
                if (best) finish(best);
                else fail(new Error('定位超时'));
            }, maxWait);
        });
    }

    global.ClickGeolocation = {
        getAccuratePosition: getAccuratePosition,
        toAmapCoordinate: toAmapCoordinate
    };
})(window);
